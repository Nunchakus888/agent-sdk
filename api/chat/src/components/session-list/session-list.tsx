import {ReactElement, useEffect, useState} from 'react';
import useFetch from '@/hooks/useFetch';
import Session from './session-list-item/session-list-item';
import {AgentInterface, PaginatedSessionsInterface, processSessionObject} from '@/utils/interfaces';
import {useAtom} from 'jotai';
import {agentAtom, agentsAtom, refreshTriggerAtom, sessionAtom, sessionsAtom} from '@/store';
import {NEW_SESSION_ID} from '../agents-list/agent-list';
import {twJoin} from 'tailwind-merge';
import {Button} from '../ui/button';

export default function SessionList({filterSessionVal}: {filterSessionVal: string}): ReactElement {
	const [editingTitle, setEditingTitle] = useState<string | null>(null);
	const [session] = useAtom(sessionAtom);
	const [page, setPage] = useState(1);
	const [pageSize] = useState(10);
	const {data, ErrorTemplate, loading, refetch} = useFetch<PaginatedSessionsInterface>('sessions', {page, page_size: pageSize}, [page]);
	const {data: agentsData} = useFetch<AgentInterface[]>('agents');
	const [, setAgents] = useAtom(agentsAtom);
	const [agent] = useAtom(agentAtom);
	const [sessions, setSessions] = useAtom(sessionsAtom);
	const [filteredSessions, setFilteredSessions] = useState(sessions);
	const [refreshTrigger] = useAtom(refreshTriggerAtom);

	useEffect(() => {
		if (agentsData) {
			setAgents(agentsData);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [agentsData]);


	// 监听全局刷新触发器
	useEffect(() => {
		if (refreshTrigger > 0) {
			setPage(1); // Reset to first page on refresh
			refetch();
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [refreshTrigger]);

	useEffect(() => {
		if (data?.items) {
			const processedSessions = data.items.map(processSessionObject);
			setSessions(processedSessions);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [data]);

	useEffect(() => {
		if (!filterSessionVal?.trim()) {
			setFilteredSessions(sessions || []);
		} else {
			setFilteredSessions((sessions || []).filter((session) => session.title?.toLowerCase()?.includes(filterSessionVal?.toLowerCase()) || session.id?.toLowerCase()?.includes(filterSessionVal?.toLowerCase())));
		}
	}, [filterSessionVal, sessions]);

	return (
		<div className={twJoin('flex flex-col h-[calc(100%-68px)] border-e')}>
			<div data-testid='sessions' className='bg-white px-[12px] border-white flex-1 fixed-scroll justify-center w-[352px] overflow-auto'>
				{loading && !sessions?.length && <div className='p-4 text-center'>loading...</div>}
				{!loading && !sessions?.length && <div className='p-4 text-center text-gray-500'>No sessions found</div>}
				{session?.id === NEW_SESSION_ID && <Session className='opacity-50' data-testid='session' isSelected={true} session={{...session, agent_id: agent?.id || '', customer_id: session?.customer_id || ''}} key={NEW_SESSION_ID} />}
				{filteredSessions.toReversed().map((s, i) => (
					<Session data-testid='session' tabIndex={sessions.length - i} editingTitle={editingTitle} setEditingTitle={setEditingTitle} isSelected={s.id === session?.id} refetch={refetch} session={s} key={s.id} />
				))}
				{ErrorTemplate && <ErrorTemplate />}
			</div>
			{data && data.total_pages > 1 && (
				<div className='flex items-center justify-between px-4 py-3 bg-white border-t'>
					<div className='text-sm text-gray-600'>
						Page {data.page} of {data.total_pages} ({data.total} total)
					</div>
					<div className='flex gap-2'>
						<Button
							size='sm'
							variant='outline'
							onClick={() => setPage(p => Math.max(1, p - 1))}
							disabled={page === 1 || loading}
						>
							Previous
						</Button>
						<Button
							size='sm'
							variant='outline'
							onClick={() => setPage(p => Math.min(data.total_pages, p + 1))}
							disabled={page === data.total_pages || loading}
						>
							Next
						</Button>
					</div>
				</div>
			)}
		</div>
	);
}
