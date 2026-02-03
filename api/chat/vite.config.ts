import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';
import { copyFileSync } from 'fs';

// https://vitejs.dev/config/
export default defineConfig(() => {
  return {
    base: '/chat/',
    test: {
      globals: true,
      environment: 'jsdom',
      includeSource: ['app/**/*.{jsx,tsx}'],
      setupFiles: ['./setupTests.ts']
    },
    plugins: [
      react(),
      {
        name: 'copy-chat-test',
        closeBundle() {
          try {
            copyFileSync(
              path.resolve(__dirname, 'chat-test.html'),
              path.resolve(__dirname, 'dist', 'chat-test.html')
            );
            console.log('✓ chat-test.html copied to dist/');
          } catch (err) {
            console.error('Failed to copy chat-test.html:', err);
          }
        }
      }
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    // 定义环境变量
    define: {
      // 开发和生产环境都使用 /api 前缀
      'import.meta.env.VITE_BASE_URL': JSON.stringify('/api'),
    },
    server: {
      port: 8002,
      host: '127.0.0.1',
      proxy: {
        // 代理所有 /api 请求到后端服务（包括 WebSocket）
        // 不重写路径，后端路由已包含 /api 前缀
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          secure: false,
          ws: true,  // 支持 WebSocket 升级
        },
        // 代理 /logs WebSocket 连接 - 使用 http:// 而不是 ws://
        '/logs': {
          target: 'http://127.0.0.1:8000',
          ws: true,
          changeOrigin: true
        },
        // 代理其他 WebSocket 连接
        '/ws': {
          target: 'http://127.0.0.1:8000',
          ws: true,
          changeOrigin: true
        }
      }
    }
  };
});
