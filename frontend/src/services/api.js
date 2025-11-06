/**
 * API 服务模块
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 创建 axios 实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60秒超时
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method, config.url);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('请求错误:', error.message);
    return Promise.reject(error);
  }
);

/**
 * 加载模型配置
 */
export const loadModel = async () => {
  const response = await apiClient.post('/api/config/load');
  return response.data;
};

/**
 * 获取配置状态
 */
export const getConfigStatus = async () => {
  const response = await apiClient.get('/api/config/status');
  return response.data;
};

/**
 * 单条语音推理
 */
export const inferenceSingle = async (audioFile) => {
  const formData = new FormData();
  formData.append('file', audioFile);

  const response = await apiClient.post('/api/inference/single', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

/**
 * 清理上传文件
 */
export const clearUploads = async () => {
  const response = await apiClient.delete('/api/uploads/clear');
  return response.data;
};

/**
 * WebSocket 连接管理
 */
export class RealtimeWebSocket {
  constructor(onMessage, onError, onClose) {
    this.ws = null;
    this.onMessage = onMessage;
    this.onError = onError;
    this.onClose = onClose;
  }

  connect() {
    const wsUrl = API_BASE_URL.replace('http', 'ws') + '/ws/realtime';
    console.log('连接 WebSocket:', wsUrl);

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('WebSocket 连接已建立');
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.onMessage(data);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket 错误:', error);
      this.onError(error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket 连接已关闭');
      this.onClose();
    };
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    } else {
      console.error('WebSocket 未连接');
    }
  }

  close() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
}

export default apiClient;
