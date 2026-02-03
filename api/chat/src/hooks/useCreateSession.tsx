import { useAtom } from 'jotai';
import { agentAtom, chatConfigAtom, customerAtom, dialogAtom, sessionAtom } from '@/store';
import { ChatConfigInterface, SessionInterface } from '@/utils/interfaces';
import { NEW_SESSION_ID } from '@/components/agents-list/agent-list';
import ChatConfigForm from '@/components/chat-config-form/chat-config-form';

export const useCreateSession = () => {
	const [, setSession] = useAtom(sessionAtom);
	const [, setAgent] = useAtom(agentAtom);
	const [, setChatConfig] = useAtom(chatConfigAtom);
	const [, setCustomer] = useAtom(customerAtom);
	const [dialog] = useAtom(dialogAtom);

	const createNewSession = () => {
		setSession(null);
		setAgent(null);
		setChatConfig(null);
		setCustomer(null); // 清理 customer 状态，避免变量污染
		
		// 弹出 ChatConfigForm 来收集配置信息
		dialog.openDialog('', 
			<ChatConfigForm 
				onSubmit={handleConfigSubmit}
				onCancel={dialog.closeDialog}
				hideBackButton={true}
			/>, 
			{height: '700px', width: '600px'}
		);
	};

	const handleConfigSubmit = (config: ChatConfigInterface) => {
		// 暂存配置信息到全局状态
		setChatConfig(config);
		
		// 创建临时会话对象（用于显示聊天界面，但还未真正创建会话）
		const tempSession: SessionInterface = {
			id: NEW_SESSION_ID,
			title: config.session_title || 'New Chat Session',
			customer_id: config.customer_id || '',
			creation_utc: new Date().toISOString(),
			mode: 'auto',
			// 保存配置信息到会话中，供后续发送消息时使用
			tenant_id: config.tenant_id,
			chatbot_id: config.chatbot_id,
			md5_checksum: config.md5_checksum,
			is_preview: config.is_preview,
			preview_action_book_ids: config.preview_action_book_ids,
			autofill_params: config.autofill_params || {'phoneNumber': '', 'tenantId': '', 'chatbotId': '', 'dialogId': '', 'keywords': ''},
			timeout: config.timeout,
		};
		
		// 设置临时会话并关闭对话框，进入聊天界面
		setSession(tempSession);
		dialog.closeDialog();
	};

	return {
		createNewSession
	};
};
