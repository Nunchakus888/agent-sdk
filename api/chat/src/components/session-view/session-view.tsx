/* eslint-disable react-hooks/exhaustive-deps */
import React, {ReactElement, useEffect, useRef, useState} from 'react';
import useFetch from '@/hooks/useFetch';
import {Textarea} from '../ui/textarea';
import {Button} from '../ui/button';
import {deleteData, postData} from '@/utils/api';
import {groupBy} from '@/utils/obj';
import Message from '../message/message';
import {EventInterface, ServerStatus, SessionInterface, processSessionObject, isCustomerSource} from '@/utils/interfaces';
import Spacer from '../ui/custom/spacer';
import {toast} from 'sonner';
import {NEW_SESSION_ID} from '../chat-header/chat-header';
import {useQuestionDialog} from '@/hooks/useQuestionDialog';
import {twJoin, twMerge} from 'tailwind-merge';
import MessageDetails from '../message-details/message-details';
import {useAtom} from 'jotai';
import {agentAtom, agentsAtom, chatConfigAtom, emptyPendingMessage, newSessionAtom, pendingMessageAtom, refreshTriggerAtom, sessionAtom, sessionsAtom, viewingMessageDetailsAtom} from '@/store';
import ErrorBoundary from '../error-boundary/error-boundary';
import DateHeader from './date-header/date-header';
// import SessoinViewHeader from './session-view-header/session-view-header';
import {getIndexedItemsFromIndexedDB, isSameDay} from '@/lib/utils';
import {DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger} from '../ui/dropdown-menu';
import {ShieldEllipsis} from 'lucide-react';
import {soundDoubleBlip} from '@/utils/sounds';

const SessionView = (): ReactElement => {
	const lastMessageRef = useRef<HTMLDivElement>(null);
	const submitButtonRef = useRef<HTMLButtonElement>(null);
	const textareaRef = useRef<HTMLTextAreaElement>(null);
	const messagesRef = useRef<HTMLDivElement>(null);
	const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const pollRemainingRef = useRef(0);

	const [message, setMessage] = useState('');
	const [lastOffset, setLastOffset] = useState(0);
	const [messages, setMessages] = useState<EventInterface[]>([]);
	const [showTyping, setShowTyping] = useState(false);
	const [showThinking, setShowThinking] = useState(false);
	const [thinkingDisplay, setThinkingDisplay] = useState('');
	const [isFirstScroll, setIsFirstScroll] = useState(true);
	const {openQuestionDialog, closeQuestionDialog} = useQuestionDialog();
	const [useContentFiltering, setUseContentFiltering] = useState(false);
	const [showLogsForMessage, setShowLogsForMessage] = useState<EventInterface | null>(null);
	const [isMissingAgent, setIsMissingAgent] = useState<boolean | null>(null);
	const [isContentFilterMenuOpen, setIsContentFilterMenuOpen] = useState(false);
	const [flaggedItems, setFlaggedItems] = useState<Record<string, string>>({});
	const [refreshFlag, setRefreshFlag] = useState(false);
	const [pendingMessage, setPendingMessage] = useAtom<EventInterface>(pendingMessageAtom);
	const [agents] = useAtom(agentsAtom);
	const [session, setSession] = useAtom(sessionAtom);
	const [agent] = useAtom(agentAtom);
	const [newSession, setNewSession] = useAtom(newSessionAtom);
	const [, setViewingMessage] = useAtom(viewingMessageDetailsAtom);
	const [, setSessions] = useAtom(sessionsAtom);
	const [chatConfig] = useAtom(chatConfigAtom);
	const [, setRefreshTrigger] = useAtom(refreshTriggerAtom);
	const {data: lastEvents, refetch, ErrorTemplate, abortFetch} = useFetch<EventInterface[]>(`sessions/${session?.id}/events`, {min_offset: lastOffset, wait_for_data: 0}, [session?.id, lastOffset], session?.id !== NEW_SESSION_ID, !!(session?.id && session?.id !== NEW_SESSION_ID), false);

	const resetChat = () => {
		setMessage('');
		setLastOffset(0);
		setMessages([]);
		setShowTyping(false);
		setShowLogsForMessage(null);
		if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
		pollRemainingRef.current = 0;
	};

	// 按需轮询：仅在等待响应时轮询，空闲时停止
	const schedulePoll = () => {
		if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
		if (pollRemainingRef.current <= 0) return;
		pollRemainingRef.current--;
		pollTimerRef.current = setTimeout(() => refetch(), 3000);
	};

	const resendMessageDialog = (index: number) => (sessionId: string, text?: string) => {
		const isLastMessage = index === messages.length - 1;
		const lastUserMessageOffset = messages[index].offset;

		if (isLastMessage) {
			setShowLogsForMessage(null);
			return resendMessage(index, sessionId, lastUserMessageOffset, text);
		}

		const onApproved = () => {
			setShowLogsForMessage(null);
			closeQuestionDialog();
			resendMessage(index, sessionId, lastUserMessageOffset, text);
		};

		const question = 'Resending this message would cause all of the following messages in the session to disappear.';
		openQuestionDialog('Are you sure?', question, [{text: 'Resend Anyway', onClick: onApproved, isMainAction: true}]);
	};

	const regenerateMessageDialog = (index: number) => (sessionId: string) => {
		const isLastMessage = index === messages.length - 1;
		const prevMessages = messages.slice(0, index + 1);
		const lastUserMessageIndex = prevMessages.findLastIndex((message) => isCustomerSource(message.source) && message.kind === 'message');
		const lastUserMessage = prevMessages[lastUserMessageIndex];
		const lastUserMessageOffset = lastUserMessage?.offset ?? messages.length - 1;

		if (isLastMessage) {
			setShowLogsForMessage(null);
			return regenerateMessage(lastUserMessageIndex, sessionId, lastUserMessageOffset);
		}

		const onApproved = () => {
			setShowLogsForMessage(null);
			closeQuestionDialog();
			regenerateMessage(lastUserMessageIndex, sessionId, lastUserMessageOffset);
		};

		const question = 'Regenerating this message would cause all of the following messages in the session to disappear.';
		openQuestionDialog('Are you sure?', question, [{text: 'Regenerate Anyway', onClick: onApproved, isMainAction: true}]);
	};

	const resendMessage = async (index: number, sessionId: string, offset: number, text?: string) => {
		const event = messages[index];

		const deleteSession = await deleteData(`sessions/${sessionId}/events?min_offset=${offset}`).catch((e) => ({error: e}));
		if (deleteSession?.error) {
			toast.error(deleteSession.error.message || deleteSession.error);
			return;
		}
		abortFetch?.();
		setLastOffset(offset);
		setMessages((messages) => messages.slice(0, index));
		postMessage(text ?? event.data?.message);
	};

	const regenerateMessage = async (index: number, sessionId: string, offset: number) => {
		resendMessage(index, sessionId, offset);
	};

	const formatMessagesFromEvents = () => {
		if (session?.id === NEW_SESSION_ID) return;
		const lastEvent = lastEvents?.at(-1);
		const lastStatusEvent = lastEvents?.findLast((e) => e.kind === 'status');
		if (!lastEvent) {
			// 空响应也消耗一次轮询配额
			schedulePoll();
			return;
		}

		// 收到新事件，重置轮询配额（保持活跃轮询直到事件流结束）
		pollRemainingRef.current = Math.max(pollRemainingRef.current, 5);

		const offset = lastEvent?.offset;
		// 收到新事件后推进 offset 游标，下次轮询从 offset+1 开始
		if ((offset || offset === 0) && offset >= lastOffset) {
			setLastOffset(offset + 1);
		}

		const correlationsMap = groupBy(lastEvents || [], (item: EventInterface) => item?.correlation_id.split('::')[0]);

		const newMessages = lastEvents?.filter((e) => e.kind === 'message') || [];
		const withStatusMessages = newMessages.map((newMessage, i) => {
			const data: EventInterface = {...newMessage};
			const item = correlationsMap?.[newMessage.correlation_id.split('::')[0]]?.at(-1)?.data;
			data.serverStatus = (item?.status || (newMessages[i + 1] ? 'ready' : null)) as ServerStatus;
			if (data.serverStatus === 'error') data.error = item?.data?.exception;
			return data;
		});

		setMessages((messages) => {
			const last = messages.at(-1);
			if (last && isCustomerSource(last.source) && correlationsMap?.[last?.correlation_id]) {
				last.serverStatus = correlationsMap[last.correlation_id].at(-1)?.data?.status || last.serverStatus;
				if (last.serverStatus === 'error') last.error = correlationsMap[last.correlation_id].at(-1)?.data?.data?.exception;
			}
			if (!withStatusMessages?.length) return [...messages];
			if (pendingMessage?.data?.message) setPendingMessage(emptyPendingMessage());

			const newVals: EventInterface[] = [];
			for (const messageArray of [messages, withStatusMessages]) {
				for (const message of messageArray) {
					newVals[message.offset] = message;
				}
			}
			return newVals.filter((message) => message);
		});

		const lastStatusEventStatus = lastStatusEvent?.data?.status;

		if (newMessages?.length && (showThinking || showTyping)) soundDoubleBlip(true);
		if (lastStatusEventStatus) {
			setShowThinking(lastStatusEventStatus === 'processing');

			if (lastStatusEventStatus === 'processing') {
				setThinkingDisplay(lastStatusEvent?.data?.data?.stage ?? 'Thinking');
			}

			setShowTyping(lastStatusEventStatus === 'typing');
		}
		// 收到事件后安排下一次轮询（倒计时递减，空闲时自动停止）
		schedulePoll();
	};

	const scrollToLastMessage = () => {
		lastMessageRef?.current?.scrollIntoView?.({behavior: isFirstScroll ? 'instant' : 'smooth'});
		if (lastMessageRef?.current && isFirstScroll) setIsFirstScroll(false);
	};

	const resetSession = () => {
		setIsFirstScroll(true);
		if (newSession && session?.id !== NEW_SESSION_ID) setNewSession(null);
		resetChat();
		textareaRef?.current?.focus();
	};

	const getSessionFlaggedItems = async () => {
		const flaggedItems = await getIndexedItemsFromIndexedDB('Parlant-flags', 'message_flags', 'sessionIndex', session?.id as string, {name: 'sessionIndex', keyPath: 'sessionId'});
		const asMap = (flaggedItems as {correlationId: string; flagValue: string; sessionId: string}[]).reduce((acc, item) => {
			acc[item.correlationId] = item.flagValue;
			return acc;
		}, {} as Record<string, string>);
		setFlaggedItems(asMap);
	};

	useEffect(() => {
		getSessionFlaggedItems();
	}, [session?.id, refreshFlag]);

	useEffect(() => setViewingMessage(showLogsForMessage), [showLogsForMessage]);
	useEffect(formatMessagesFromEvents, [lastEvents]);
	useEffect(scrollToLastMessage, [messages?.length, pendingMessage, isFirstScroll]);
	useEffect(resetSession, [session?.id]);
	useEffect(() => {
		if (showThinking || showTyping) lastMessageRef?.current?.scrollIntoView({behavior: 'smooth'});
	}, [showThinking, showTyping]);
	useEffect(() => {
		if (agents && agent?.id) setIsMissingAgent(!agents?.find((a) => a.id === agent?.id));
	}, [agents, agent?.id]);

	// 进入会话时给几次初始轮询，捕获可能正在进行的事件
	useEffect(() => {
		if (!session?.id || session?.id === NEW_SESSION_ID) return;
		pollRemainingRef.current = 3;
		return () => {
			if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
		};
	}, [session?.id]);

	// Session creation now handled via /chat API in postMessage

	const postMessage = async (content: string): Promise<void> => {
		setPendingMessage((pendingMessage) => ({...pendingMessage, sessionId: session?.id, data: {message: content}}));
		setMessage('');
		
		console.log('----session', session);
    // Use /chat API for all messages
		const isNewSession = session?.id === NEW_SESSION_ID;
		
		try {
			let chatRequest: any;
			
			if (isNewSession) {
				// New session: use chatConfig or session data
				// IMPORTANT: In backend, agent_id from session is actually the chatbot_id
				chatRequest = {
					message: content,
					// Use session_id from chatConfig if provided, otherwise let backend generate one
					session_id: chatConfig?.session_id,
					// Required fields - fallback chain: chatConfig -> session
					tenant_id: chatConfig?.tenant_id || session?.tenant_id || '',
					chatbot_id: chatConfig?.chatbot_id || session?.chatbot_id || '',
					// Session info (Note: agent_id is NOT in ChatRequestDTO, it's determined server-side)
					customer_id: chatConfig?.customer_id,
					session_title: session?.title || chatConfig?.session_title || 'New Conversation',
					// Optional config fields
					md5_checksum: chatConfig?.md5_checksum || session?.md5_checksum,
					is_preview: chatConfig?.is_preview ?? session?.is_preview ?? false,
					preview_action_book_ids: chatConfig?.preview_action_book_ids || session?.preview_action_book_ids || [],
					autofill_params: chatConfig?.autofill_params || session?.autofill_params || {},
					timeout: chatConfig?.timeout || session?.timeout || 57,
				};
				
				// Remove undefined fields
				Object.keys(chatRequest).forEach(key => {
					if (chatRequest[key] === undefined) {
						delete chatRequest[key];
					}
				});
				
				console.log('📤 Sending first message (new session):', chatRequest);
			} else {
				// Existing session: get all parameters from session object
				// IMPORTANT: Use chatbot_id directly from session object (parsed by backend API)
				chatRequest = {
					message: content,
					session_id: session?.id,
					// Required fields - all from session object
					tenant_id: session?.tenant_id || '',
					chatbot_id: session?.chatbot_id || '',  // Use chatbot_id directly from API
					// Optional fields from session
					customer_id: session?.customer_id,
					md5_checksum: session?.md5_checksum,
					is_preview: session?.is_preview,
					preview_action_book_ids: session?.preview_action_book_ids,
					autofill_params: session?.autofill_params,
					timeout: session?.timeout,
				};
				
				// Remove undefined fields
				Object.keys(chatRequest).forEach(key => {
					if (chatRequest[key] === undefined) {
						delete chatRequest[key];
					}
				});
				
				console.log('📤 Sending message (existing session):', chatRequest);
				console.log('🔍 Session object:', session);
			}
			
			// Validate required fields before sending
			if (!chatRequest.chatbot_id) {
				const missing = [];
				// if (!chatRequest.tenant_id) missing.push('tenant_id');
				if (!chatRequest.chatbot_id) missing.push('chatbot_id');
				const errorMsg = `缺少必填参数: ${missing.join(', ')}。会话对象中缺少必要信息，请检查会话数据。`;
				console.error('❌ Missing required parameters:', missing);
				console.error('❌ ChatRequest:', chatRequest);
				console.error('❌ Session object:', session);
				toast.error(errorMsg);
				return;
			}
			
			const response: any = await postData('sessions/chat_async', chatRequest);
			console.log('✅ Chat API response:', response);
			
			// Check for error response (accept 2xx status codes)
			if (response.status >= 400) {
				const errorMsg = response.message || 'Chat request failed';
				console.error('❌ Chat API error response:', response);
				toast.error(`发送失败: ${errorMsg}`);
				return;
			}

			if (isNewSession && response.session_id) {
				// Update session with actual session_id from backend, preserve all config
				const actualSession: SessionInterface = processSessionObject({
					...session!,
					id: response.session_id,
					// Preserve config for future messages
					tenant_id: chatRequest.tenant_id,
					chatbot_id: chatRequest.chatbot_id,
					md5_checksum: chatRequest.md5_checksum,
					is_preview: chatRequest.is_preview,
					preview_action_book_ids: chatRequest.preview_action_book_ids,
					autofill_params: chatRequest.autofill_params,
					timeout: chatRequest.timeout,
				});
				setSession(actualSession);
				setNewSession(null);
				// Keep chatConfig for potential future use
				// setChatConfig(null);
				
				// Add session to sessions list
				setSessions((sessions) => [actualSession, ...sessions]);
				
				// Trigger global refresh to update session list in sidebar
				setRefreshTrigger(prev => prev + 1);
			}
			
			soundDoubleBlip();
			// 发送消息后启动轮询（最多 20 次 × 3s = 60s），等待 assistant 回复
			pollRemainingRef.current = 20;
			refetch();
		} catch (err: any) {
			console.error('❌ Chat API error:', err);
			const errorMsg = err?.message || err?.data?.message || '发送消息失败';
			toast.error(`错误: ${errorMsg}`);
		}
	};

	const handleTextareaKeydown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submitButtonRef?.current?.click();
		} else if (e.key === 'Enter' && e.shiftKey) e.preventDefault();
	};

	const isCurrSession = (session?.id === NEW_SESSION_ID && !pendingMessage?.id) || (session?.id !== NEW_SESSION_ID && pendingMessage?.sessionId === session?.id);
	const visibleMessages = (!messages?.length || isCurrSession) && pendingMessage?.data?.message ? [...messages, pendingMessage] : messages;

	const showLogs = (i: number) => (event: EventInterface) => {
		event.index = i;
		setShowLogsForMessage(event.id === showLogsForMessage?.id ? null : event);
	};

	return (
		<>
			<div ref={messagesRef} className={twMerge('flex items-center h-full w-full bg-white gap-[14px] rounded-[10px]', showLogsForMessage && 'bg-green-light')}>
				<div className={twMerge('h-full w-full pb-[14px] pt-[10px] rounded-[10px] flex flex-col transition-all duration-500 bg-white', showLogsForMessage && 'w-[calc(100%-min(700px,35vw))]')}>
					<div className='h-full flex flex-col rounded-[10px] m-auto w-full min-w-[unset]'>
						{/* <div className='h-[58px] bg-[#f5f5f9]'></div> */}
						{/* <SessoinViewHeader /> */}
						{/* <div className={twMerge('h-[21px] border-t-0 bg-white')}></div> */}
						<div className={twMerge('flex flex-col rounded-es-[16px] rounded-ee-[16px] items-center bg-white mx-auto w-full flex-1 overflow-hidden')}>
							<div
								className={twJoin(
									'messages fixed-scroll flex-1 flex flex-col w-full pb-4 overflow-x-hidden'
									// '[scroll-snap-type:y_mandatory]'
								)}
								aria-live='polite'
								role='log'
								aria-label='Chat messages'>
								{ErrorTemplate && <ErrorTemplate />}
								{visibleMessages.map((event, i) => (
									<React.Fragment key={(event.correlation_id || 0) + `${i}`}>
										{!isSameDay(messages[i - 1]?.creation_utc, event.creation_utc) && <DateHeader date={event.creation_utc} isFirst={!i} bgColor='bg-white' />}
										<div ref={lastMessageRef} className='flex snap-end flex-col max-w-[min(1020px,100%)] w-[1020px] self-center'>
											<Message
												flaggedChanged={() => {
													setRefreshFlag((val) => !val);
												}}
												flagged={flaggedItems[event.correlation_id]}
												isFirstMessageInDate={!isSameDay(messages[i - 1]?.creation_utc, event.creation_utc)}
												isRegenerateHidden={!!isMissingAgent}
												event={event}
												sameCorrelationMessages={visibleMessages.filter((e) => e.correlation_id === event.correlation_id)}
												isContinual={event.correlation_id === visibleMessages[i - 1]?.correlation_id || (isCustomerSource(event.source) && visibleMessages[i - 1] && isCustomerSource(visibleMessages[i - 1].source))}
												regenerateMessageFn={regenerateMessageDialog(i)}
												resendMessageFn={resendMessageDialog(i)}
												showLogsForMessage={showLogsForMessage}
												showLogs={showLogs(i)}
											/>
										</div>
									</React.Fragment>
								))}
								{(showTyping || showThinking) && (
									<div ref={lastMessageRef} className='flex snap-end max-w-[min(1020px,100%)] w-[1020px] self-center'>
										<div className='bubblesWrapper snap-end' aria-hidden='true'>
											<div className='bubbles' />
										</div>
										{showTyping && <p className={twMerge('flex items-center font-normal text-[#A9AFB7] text-[14px] font-inter')}>Typing...</p>}
										{showThinking && <p className={twMerge('flex items-center font-normal text-[#A9AFB7] text-[14px] font-inter')}>{thinkingDisplay}...</p>}
									</div>
								)}
							</div>
							<div className={twMerge('w-full flex justify-between', isMissingAgent && 'hidden')}>
								<Spacer />
								<div className='group relative border flex-1 border-muted border-solid rounded-[10px] flex flex-row justify-center items-center bg-white p-[0.9rem] ps-[14px] pe-0 h-[48.67px] max-w-[1000px] mb-[26px]'>
									<DropdownMenu open={isContentFilterMenuOpen} onOpenChange={setIsContentFilterMenuOpen}>
										<DropdownMenuTrigger className='outline-none' data-testid='menu-button' tabIndex={-1} onClick={(e) => e.stopPropagation()}>
											<div className={twMerge('me-[2px] border border-transparent hover:bg-[#F3F5F9] rounded-[6px] size-[25px] flex items-center justify-center', isContentFilterMenuOpen && '!bg-[#f5f6f8]')}>
												{!useContentFiltering && <img src='icons/edit.svg' alt='' className={twMerge('h-[14px] w-[14px]')} />}
												{useContentFiltering && <ShieldEllipsis className={twJoin('size-[18px]')} />}
											</div>
										</DropdownMenuTrigger>
										<DropdownMenuContent side='top' align='start' className='max-w-[480px] -ms-[10px] flex flex-col gap-[8px] py-[14px] px-[10px] border-none [box-shadow:_0px_8px_20px_-8px_#00000012] rounded-[8px]'>
											<DropdownMenuItem
												tabIndex={0}
												onClick={() => setUseContentFiltering(false)}
												className={twMerge('gap-0  cursor-pointer font-normal text-[14px] px-[10px] font-inter capitalize hover:!bg-[#FAF9FF]', !useContentFiltering && '!bg-[#f5f6f8] hover:!bg-[#f5f6f8]')}>
												<img src='icons/edit.svg' alt='' className={twMerge('me-[8px] size-[15px]')} />
												Direct (No Moderation)
											</DropdownMenuItem>
											<DropdownMenuItem
												tabIndex={0}
												onClick={() => setUseContentFiltering(true)}
												className={twMerge('gap-0 !cursor-pointer font-normal text-[14px] items-start px-[10px] font-inter  hover:!bg-[#FAF9FF]', useContentFiltering && '!bg-[#f5f6f8] hover:!bg-[#f5f6f8]')}>
												<ShieldEllipsis className='me-[8px] !size-[17px] mt-[3px]' />
												<div>
													<div>Content Moderation</div>
													<small className='font-light'>
														Messages will be flagged for harmful or illicit content and censored accordingly. The agent will see such messages were sent and the reason why they were censored, but it won't see their content.
													</small>
												</div>
											</DropdownMenuItem>
										</DropdownMenuContent>
									</DropdownMenu>
									<Textarea
										role='textbox'
										ref={textareaRef}
										placeholder='Message...'
										value={message}
										onKeyDown={handleTextareaKeydown}
										onChange={(e) => setMessage(e.target.value)}
										rows={1}
										className='box-shadow-none placeholder:text-[#282828] resize-none border-none h-full rounded-none min-h-[unset] p-0 whitespace-nowrap no-scrollbar font-inter font-light text-[16px] leading-[100%] bg-white'
									/>
								<Button 
									variant='ghost' 
									data-testid='submit-button' 
									className='max-w-[60px] rounded-full hover:bg-white' 
									ref={submitButtonRef} 
									disabled={!message?.trim()} 
									onClick={() => postMessage(message)}
								>
									<img src='icons/send.svg' alt='Send' height={19.64} width={21.52} className='h-10' />
								</Button>
								</div>
								<Spacer />
							</div>
							<div className='w-full'>
								<Spacer />
								<div></div>
								<Spacer />
							</div>
						</div>
					</div>
				</div>
				<ErrorBoundary component={<div className='flex h-full min-w-[50%] justify-center items-center text-[20px]'>Failed to load logs</div>}>
					<div
						className={twMerge(
							'fixed top-0 left-[unset] h-full right-0 z-[99] bg-white translate-x-[100%] max-w-[min(700px,35vw)] [box-shadow:0px_0px_30px_0px_#0000001F] w-[min(700px,35vw)] [transition-duration:600ms]',
							showLogsForMessage && 'translate-x-0'
						)}>
						{showLogsForMessage && (
							<MessageDetails
								flaggedChanged={() => {
									setRefreshFlag((val) => !val);
								}}
								sameCorrelationMessages={visibleMessages.filter((e) => e.correlation_id === showLogsForMessage.correlation_id)}
								event={showLogsForMessage}
								regenerateMessageFn={showLogsForMessage?.index ? regenerateMessageDialog(showLogsForMessage.index) : undefined}
								resendMessageFn={showLogsForMessage?.index || showLogsForMessage?.index === 0 ? resendMessageDialog(showLogsForMessage.index) : undefined}
								closeLogs={() => setShowLogsForMessage(null)}
							/>
						)}
					</div>
				</ErrorBoundary>
			</div>
		</>
	);
};

export default SessionView;
