/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable react-hooks/exhaustive-deps */
import {EventInterface, Log} from '@/utils/interfaces';
import React, {memo, ReactNode, useEffect, useRef, useState} from 'react';
import {getMessageLogs, getMessageLogsWithFilters} from '@/utils/logs';
import {twJoin, twMerge} from 'tailwind-merge';
import clsx from 'clsx';
import {useLocalStorage} from '@/hooks/useLocalStorage';
import LogFilters, {Level, Type} from '../log-filters/log-filters';
import CannedResponses from '../canned-responses/canned-responses';
import EmptyState from './empty-state';
import FilterTabs from './filter-tabs';
import MessageDetailsHeader from './message-details-header';
import {ResizableHandle, ResizablePanel, ResizablePanelGroup} from '../ui/resizable';
import {ImperativePanelHandle} from 'react-resizable-panels';
import Tooltip from '../ui/custom/tooltip';
import {copy} from '@/lib/utils';
import MessageLogs from './message-logs';
import CopyText from '../ui/custom/copy-text';
import {useAtom} from 'jotai';
import {sessionAtom} from '@/store';

interface DefInterface {
	level?: Level;
	types?: Type[];
	content?: string[];
}

interface Filter {
	id: number;
	selected?: boolean;
	name: string;
	def: DefInterface | null;
}

const MessageError = ({event}: {event: EventInterface}) => {
	return (
		<div className='h-full group p-[20px] bg-[#ebecf0] text-[13px] text-[#ef5350] z-10'>
			<pre className={clsx('p-[10px] max-w-[-webkit-fill-available] pe-[10px] text-wrap break-words bg-white rounded-[8px] h-full overflow-auto  group relative')}>
				<div className='sticky h-0 hidden z-10 group-hover:block [direction:rtl] top-[10px] right-[10px] gap-[10px]'>
					<Tooltip value='Copy' side='top'>
						<img src='icons/copy.svg' sizes='18' alt='' onClick={() => copy(event?.error || '')} className='cursor-pointer' />
					</Tooltip>
				</div>
				{event?.error}
			</pre>
		</div>
	);
};

const getDefaultSelectedActiveTab = (filterTabs: Filter[]) => {
	return filterTabs.find((t) => t.selected)?.id || filterTabs[0]?.id || null;
};

const MessageDetails = ({
	event,
	closeLogs,
	regenerateMessageFn,
	resendMessageFn,
	flaggedChanged,
	sameCorrelationMessages,
}: {
	event?: EventInterface | null;
	sameCorrelationMessages?: EventInterface[];
	closeLogs?: VoidFunction;
	regenerateMessageFn?: (sessionId: string) => void;
	resendMessageFn?: (sessionId: string) => void;
	flaggedChanged?: (flagged: boolean) => void;
}): ReactNode => {
	const [filters, setFilters] = useState<Record<string, any> | null>(null);
	const [filterTabs, setFilterTabs] = useLocalStorage<Filter[]>('filters', []);
	const [currFilterTabs, setCurrFilterTabs] = useState<number | null>(getDefaultSelectedActiveTab(filterTabs as Filter[]));
	const [logs, setLogs] = useState<Log[] | null>(null);
	const [filteredLogs, setFilteredLogs] = useState<Log[]>([]);
	const messagesRef = useRef<HTMLDivElement | null>(null);
	const resizableRef = useRef<ImperativePanelHandle | null>(null);
	
	// Get session data
	const [session] = useAtom(sessionAtom);

	useEffect(() => {
		(setFilterTabs as React.Dispatch<React.SetStateAction<Filter[]>>)((prev) => {
			const newTabs = prev.map((tab: Filter) => {
				tab.selected = tab.id === currFilterTabs;
				return tab;
			});
			return newTabs;
		});
	}, [currFilterTabs]);

	useEffect(() => {
		if (event?.id) resizableRef.current?.resize(50);
	}, [event?.id]);

	useEffect(() => {
		const setLogsFn = async () => {
			const hasFilters = Object.keys(filters || {}).length;
			if (logs && filters) {
				if (!hasFilters && filters) setFilteredLogs(logs);
				else {
					const filtered = await getMessageLogsWithFilters(event?.correlation_id as string, filters as {level: string; types?: string[]; content?: string[]});
					setFilteredLogs(filtered);
					(setFilterTabs as React.Dispatch<React.SetStateAction<Filter[]>>)((tabFilters: Filter[]) => {
						if (!tabFilters.length && hasFilters) {
							const filter = {id: Date.now(), def: filters, name: 'Logs'};
							setCurrFilterTabs(filter.id);
							return [filter];
						}
						const tab = tabFilters.find((t) => t.id === currFilterTabs);
						if (!tab) return tabFilters;
						tab.def = filters;
						return [...tabFilters];
					});
				}
			}
			if (!filters && logs?.length) {
				setFilters({});
			}
		};
		setLogsFn();
	}, [logs, filters]);

	useEffect(() => {
		if (!event && logs) {
			setLogs(null);
			setFilteredLogs([]);
		}
	}, [event]);

	useEffect(() => {
		if (!event?.correlation_id) return;
		console.log('Fetching logs for correlation_id:', event.correlation_id);
		const setLogsFn = async () => {
			try {
				const logs = await getMessageLogs(event.correlation_id);
				console.log('Logs fetched:', logs?.length || 0, 'items');
				setLogs(logs);
			} catch (error) {
				console.error('Error fetching logs:', error);
				setLogs([]);  // Set to empty array on error
			}
		};
		setLogsFn();
	}, [event?.correlation_id]);

	const deleteFilterTab = (id: number | undefined) => {
		const filterIndex = (filterTabs as Filter[]).findIndex((t) => t.id === id);
		if (filterIndex === -1) return;
		const filteredTabs = (filterTabs as Filter[]).filter((t) => t.id !== id);
		(setFilterTabs as any)(filteredTabs);

		if (currFilterTabs === id) {
			const newTab = filteredTabs?.[(filterIndex || 1) - 1]?.id || filteredTabs?.[0]?.id || null;
			setCurrFilterTabs(newTab);
		}
		if (!filteredTabs.length) setFilters({});
	};

	const shouldRenderTabs = event && !!logs?.length && !!filterTabs?.length;
	const showCannedResponse = false;
	const cannedResponseEntries = Object.entries(event?.data?.canned_responses || {}).map(([id, value]) => ({id, value}));
	const isError = event?.serverStatus === 'error';

  // Use chatbot_id directly from session object (processed by processSessionObject)
  const chatbotId = session?.chatbot_id;

	return (
		<div className={twJoin('w-full h-full animate-fade-in duration-200 overflow-auto flex flex-col justify-start pt-0 pe-0 bg-[#FBFBFB]')}>
			<MessageDetailsHeader
				event={event || null}
				closeLogs={closeLogs}
				sameCorrelationMessages={sameCorrelationMessages}
				resendMessageFn={resendMessageFn}
				regenerateMessageFn={regenerateMessageFn}
				className={twJoin('shadow-main h-[60px] min-h-[60px]', Object.keys(filters || {}).length ? 'border-[#F3F5F9]' : '')}
				flaggedChanged={flaggedChanged}
			/>
			<div className='ps-[20px] pt-[10px] pb-[10px] flex flex-col gap-[8px] bg-white border-b border-[#F3F5F9]'>
				<div className='flex items-center gap-[3px] text-[14px] font-normal'>
					<CopyText textToCopy={event?.correlation_id?.split('::')?.[0]} preText='Correlation ID: ' text={`${event?.correlation_id?.split('::')?.[0]}`} className='whitespace-nowrap [&_span]:text-ellipsis [&_span]:overflow-hidden [&_span]:block' />
				</div>
			</div>
			
			{/* Session Information Section */}
			{session && (
				<div className='ps-[20px] pt-[10px] pb-[10px] flex flex-col gap-[8px] bg-[#F8F9FA] border-b border-[#F3F5F9]'>
					<div className='text-[14px] font-semibold text-[#333] mb-[4px]'>Session Information</div>
					{session?.id && (
						<div className='flex items-center gap-[3px] text-[14px] font-normal text-[#656565]'>
							<span className='font-medium'>Session ID:</span> 
							<CopyText 
								textToCopy={session.id} 
								text={session.id} 
								className='font-mono text-[13px] cursor-pointer hover:text-[#333]' 
							/>
						</div>
					)}
					{chatbotId && (
						<div className='flex items-center gap-[3px] text-[14px] font-normal text-[#656565]'>
							<span className='font-medium'>Chatbot ID:</span> 
							<CopyText 
								textToCopy={chatbotId} 
								text={chatbotId} 
								className='font-mono text-[13px] cursor-pointer hover:text-[#333]' 
							/>
						</div>
					)}
					{session?.tenant_id && (
						<div className='flex items-center gap-[3px] text-[14px] font-normal text-[#656565]'>
							<span className='font-medium'>Tenant ID:</span> 
							<CopyText 
								textToCopy={session.tenant_id} 
								text={session.tenant_id} 
								className='font-mono text-[13px] cursor-pointer hover:text-[#333]' 
							/>
						</div>
					)}
					{session?.customer_id && (
						<div className='flex items-center gap-[3px] text-[14px] font-normal text-[#656565]'>
							<span className='font-medium'>Customer ID:</span> 
							<CopyText 
								textToCopy={session.customer_id} 
								text={session.customer_id} 
								className='font-mono text-[13px] cursor-pointer hover:text-[#333]' 
							/>
						</div>
					)}
					{session?.mode && (
						<div className='flex items-center gap-[3px] text-[14px] font-normal text-[#656565]'>
							<span className='font-medium'>Mode:</span> <span className='capitalize'>{session.mode}</span>
						</div>
					)}
					{session?.creation_utc && (
						<div className='flex items-center gap-[3px] text-[14px] font-normal text-[#656565]'>
							<span className='font-medium'>Created:</span> <span className='text-[13px]'>{new Date(session.creation_utc).toLocaleString()}</span>
						</div>
					)}
					{session?.is_preview !== undefined && (
						<div className='flex items-center gap-[3px] text-[14px] font-normal text-[#656565]'>
							<span className='font-medium'>Preview Mode:</span> <span className='text-[13px]'>{session.is_preview ? 'Yes' : 'No'}</span>
						</div>
					)}
					{session?.timeout && (
						<div className='flex items-center gap-[3px] text-[14px] font-normal text-[#656565]'>
							<span className='font-medium'>Timeout:</span> <span className='text-[13px]'>{session.timeout}ms</span>
						</div>
					)}
				</div>
			)}
			<ResizablePanelGroup direction='vertical' className={twJoin('w-full h-full overflow-auto flex flex-col justify-start pt-0 pe-0 bg-[#FBFBFB]')}>
				<ResizablePanel ref={resizableRef} minSize={0} maxSize={isError ? 99 : 0} defaultSize={isError ? 50 : 0}>
					{isError && <MessageError event={event} />}
				</ResizablePanel>
				<ResizableHandle withHandle className={twJoin(!isError && 'hidden')} />
				<ResizablePanel minSize={isError ? 0 : 100} maxSize={isError ? 99 : 100} defaultSize={isError ? 50 : 100} className='flex flex-col bg-white'>
					{showCannedResponse && !!cannedResponseEntries.length && <CannedResponses cannedResponses={cannedResponseEntries} />}
					<div className={twMerge('flex justify-between bg-white z-[1] items-center min-h-[58px] h-[58px] p-[10px] pb-[4px] pe-0', shouldRenderTabs && 'min-h-0 h-0')}>
						{!shouldRenderTabs && (
							<LogFilters
								showDropdown
								filterId={currFilterTabs || undefined}
								def={structuredClone((filterTabs as Filter[]).find((t: Filter) => currFilterTabs === t.id)?.def || null)}
								applyFn={(types, level, content) => {
									setTimeout(() => setFilters({types, level, content}), 0);
								}}
							/>
						)}
					</div>
					{shouldRenderTabs && <FilterTabs currFilterTabs={currFilterTabs} filterTabs={filterTabs as Filter[]} setFilterTabs={setFilterTabs as any} setCurrFilterTabs={setCurrFilterTabs} />}
					{event && !!logs?.length && shouldRenderTabs && (
						<LogFilters
							showTags
							showDropdown
							deleteFilterTab={deleteFilterTab}
							className={twMerge(!filteredLogs?.length && '', !logs?.length && 'absolute')}
							filterId={currFilterTabs || undefined}
							def={structuredClone((filterTabs as Filter[]).find((t: Filter) => currFilterTabs === t.id)?.def || null)}
							applyFn={(types, level, content) => {
								setTimeout(() => setFilters({types, level, content}), 0);
							}}
						/>
					)}
					{!event && <EmptyState title='Feeling curious?' subTitle='Select a message for additional actions and information about its process.' />}
					{event && logs && !logs?.length && (
						<EmptyState
							imgClassName='w-[68px] h-[48px]'
							imgUrl='logo-muted.svg'
							title='Whoopsie!'
							subTitle="The logs for this message weren't found in cache. Try regenerating it to get fresh logs."
							className={twJoin(isError && 'translate-y-[0px]')}
						/>
					)}
					{event && !!logs?.length && !filteredLogs.length && <EmptyState title='No logs for the current filters' className={twJoin(isError && 'translate-y-[0px]')} />}
					{event && !!filteredLogs.length && (
						<div className='ps-[10px] overflow-auto h-[-webkit-fill-available]'>
							<MessageLogs messagesRef={messagesRef} filteredLogs={filteredLogs} />
						</div>
					)}
				</ResizablePanel>
			</ResizablePanelGroup>
		</div>
	);
};

export default memo(MessageDetails, (prev, next) => prev.event === next.event);
