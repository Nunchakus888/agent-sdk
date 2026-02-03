import {useState, useCallback} from 'react';
import {postData} from '@/utils/api';
import {ChatRequestInterface, ChatResponseInterface} from '@/utils/interfaces';

interface UseChatAPIReturn {
	sendMessage: (request: ChatRequestInterface) => Promise<ChatResponseInterface>;
	loading: boolean;
	error: string | null;
}

/**
 * Custom hook for interacting with the /chat API endpoint
 * Handles message sending with automatic session and customer management
 */
export const useChatAPI = (): UseChatAPIReturn => {
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const sendMessage = useCallback(async (request: ChatRequestInterface): Promise<ChatResponseInterface> => {
		setLoading(true);
		setError(null);

		try {
			const response = await postData('sessions/chat_async', request);
			
			// Check if response indicates an error
			if (response.status !== 200) {
				throw new Error(response.message || 'Failed to send message');
			}

			return response as ChatResponseInterface;
		} catch (err) {
			const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
			setError(errorMessage);
			throw err;
		} finally {
			setLoading(false);
		}
	}, []);

	return {
		sendMessage,
		loading,
		error,
	};
};

