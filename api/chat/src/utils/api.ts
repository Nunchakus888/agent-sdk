export const BASE_URL = import.meta.env.VITE_BASE_URL || '';

/**
 * API 路径映射
 * 将前端简化路径映射到后端实际路径
 */
const API_PATH_MAP: Record<string, string> = {
	'sessions': 'v1/session',
	'sessions/chat_async': 'v1/chat',  // chat_async maps to v1/chat endpoint
	'sessions/query': 'v1/chat',  // query also maps to v1/chat endpoint
	'agents': 'v1/agent',
};

/**
 * 转换 API 路径
 * 例如: "sessions" -> "v1/session"
 *       "sessions/xxx/events" -> "v1/session/xxx/events"
 */
const mapApiPath = (endpoint: string): string => {
	// 检查是否匹配映射规则
	for (const [prefix, mapped] of Object.entries(API_PATH_MAP)) {
		if (endpoint === prefix || endpoint.startsWith(`${prefix}/`)) {
			return endpoint.replace(prefix, mapped);
		}
	}
	return endpoint;
};

/**
 * Encodes path segments in URL endpoint
 * Example: "sessions/test:id/events" -> "sessions/test%3Aid/events"
 */
const encodeEndpoint = (endpoint: string): string => {
	// Split by / and encode each segment, then rejoin
	return endpoint.split('/').map(segment => {
		// Don't encode empty segments or query strings
		if (!segment || segment.includes('?') || segment.includes('=')) {
			return segment;
		}
		// Check if segment needs encoding (contains special characters)
		if (segment !== encodeURIComponent(segment)) {
			return encodeURIComponent(segment);
		}
		return segment;
	}).join('/');
};

const request = async (url: string, options: RequestInit = {}) => {
	try {
		const response = await fetch(url, options);
		if (!response.ok) {
			throw new Error(`HTTP error! Status: ${response.status}`);
		}
		if (options.method === 'PATCH' || options.method === 'DELETE') return;
		return await response.json();
	} catch (error) {
		console.error('Fetch error:', error);
		throw error;
	}
};

export const getData = async (endpoint: string) => {
	const mappedEndpoint = mapApiPath(endpoint);
	const encodedEndpoint = encodeEndpoint(mappedEndpoint);
	return request(`${BASE_URL}/${encodedEndpoint}`);
};

export const postData = async (endpoint: string, data?: object) => {
	const mappedEndpoint = mapApiPath(endpoint);
	const encodedEndpoint = encodeEndpoint(mappedEndpoint);
	return request(`${BASE_URL}/${encodedEndpoint}`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(data),
	});
};

export const patchData = async (endpoint: string, data: object) => {
	const mappedEndpoint = mapApiPath(endpoint);
	const encodedEndpoint = encodeEndpoint(mappedEndpoint);
	return request(`${BASE_URL}/${encodedEndpoint}`, {
		method: 'PATCH',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(data),
	});
};

export const deleteData = async (endpoint: string) => {
	const mappedEndpoint = mapApiPath(endpoint);
	const encodedEndpoint = encodeEndpoint(mappedEndpoint);
	return request(`${BASE_URL}/${encodedEndpoint}`, {
		method: 'DELETE',
	});
};
