import axios, { AxiosInstance, AxiosResponse } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('[API Error]', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Generic methods
  async get<T>(url: string): Promise<T> {
    const response: AxiosResponse<T> = await this.client.get(url);
    return response.data;
  }

  async post<T>(url: string, data: any): Promise<T> {
    const response: AxiosResponse<T> = await this.client.post(url, data);
    return response.data;
  }

  async put<T>(url: string, data: any): Promise<T> {
    const response: AxiosResponse<T> = await this.client.put(url, data);
    return response.data;
  }

  async delete<T>(url: string): Promise<T> {
    const response: AxiosResponse<T> = await this.client.delete(url);
    return response.data;
  }

  // Algorithm-specific methods
  async trainAlgorithm(
    category: string,
    algorithmSlug: string,
    params: any
  ): Promise<any> {
    // Special case for sentiment analysis which uses /analyze endpoint
    const endpoint = algorithmSlug === 'sentiment-analysis' ? 'analyze' : 'train';
    return this.post(`/api/${category}/${algorithmSlug}/${endpoint}`, params);
  }

  async getAlgorithmInfo(
    category: string,
    algorithmSlug: string
  ): Promise<any> {
    return this.get(`/api/${category}/${algorithmSlug}/info`);
  }

  async getAlgorithms(category: string): Promise<any[]> {
    return this.get(`/api/${category}/algorithms`);
  }

  async listAlgorithms(category: string): Promise<any[]> {
    return this.get(`/api/${category}/algorithms`);
  }

  async healthCheck(): Promise<{ status: string }> {
    return this.get('/health');
  }
}

export const apiService = new ApiService();
export default apiService;
