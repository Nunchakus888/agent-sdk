from urllib.parse import quote_plus

def encode_mongodb_url(url: str) -> str:
    """
    对MongoDB连接URL进行编码处理，处理特殊字符如+号等
    
    Args:
        url: 原始MongoDB连接URL
        
    Returns:
        编码后的MongoDB连接URL
    """
    if not url or not isinstance(url, str):
        return url
    
    # 解析URL组件
    if '://' not in url:
        return url
    
    protocol, rest = url.split('://', 1)
    
    # 检查是否包含用户名密码
    if '@' in rest:
        # 分离认证信息和主机部分
        auth_part, host_part = rest.split('@', 1)
        
        # 检查认证部分是否包含用户名密码
        if ':' in auth_part:
            username, password = auth_part.split(':', 1)
            # 对用户名和密码进行编码
            encoded_username = quote_plus(username)
            encoded_password = quote_plus(password)
            encoded_auth = f"{encoded_username}:{encoded_password}"
        else:
            encoded_auth = quote_plus(auth_part)
        
        return f"{protocol}://{encoded_auth}@{host_part}"
    else:
        # 没有认证信息，直接返回原URL
        return url
