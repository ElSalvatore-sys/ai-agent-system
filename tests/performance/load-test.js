import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('error_rate');
const responseTime = new Trend('response_time');
const requestCount = new Counter('request_count');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },  // Ramp up to 10 users
    { duration: '5m', target: 10 },  // Stay at 10 users
    { duration: '2m', target: 20 },  // Ramp up to 20 users
    { duration: '5m', target: 20 },  // Stay at 20 users
    { duration: '2m', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    http_req_failed: ['rate<0.1'],     // Error rate under 10%
    error_rate: ['rate<0.1'],          // Custom error rate under 10%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test data
const TEST_PROMPTS = [
  'Hello, how are you?',
  'What is machine learning?',
  'Explain quantum computing in simple terms.',
  'Write a short story about a robot.',
  'What are the benefits of renewable energy?',
];

const TEST_MODELS = [
  'llama2-7b',
  'codellama-13b',
  'mistral-7b'
];

export default function () {
  // Test 1: Health check
  testHealthCheck();
  
  // Test 2: Model listing
  testModelListing();
  
  // Test 3: OpenAI API compatibility
  testOpenAIAPI();
  
  // Test 4: GraphQL API
  testGraphQLAPI();
  
  // Test 5: WebSocket connection
  testWebSocket();
  
  sleep(1);
}

function testHealthCheck() {
  const response = http.get(`${BASE_URL}/health`);
  
  check(response, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  requestCount.add(1);
  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);
}

function testModelListing() {
  const response = http.get(`${BASE_URL}/v1/models`);
  
  check(response, {
    'models endpoint status is 200': (r) => r.status === 200,
    'models response has data': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.data && Array.isArray(data.data);
      } catch {
        return false;
      }
    },
  });
  
  requestCount.add(1);
  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);
}

function testOpenAIAPI() {
  const prompt = TEST_PROMPTS[Math.floor(Math.random() * TEST_PROMPTS.length)];
  const model = TEST_MODELS[Math.floor(Math.random() * TEST_MODELS.length)];
  
  const payload = JSON.stringify({
    model: model,
    messages: [
      { role: 'user', content: prompt }
    ],
    max_tokens: 100,
    temperature: 0.7
  });
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  const response = http.post(`${BASE_URL}/v1/chat/completions`, payload, params);
  
  check(response, {
    'chat completion status is 200': (r) => r.status === 200,
    'chat completion has response': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.choices && data.choices.length > 0;
      } catch {
        return false;
      }
    },
    'chat completion response time < 10s': (r) => r.timings.duration < 10000,
  });
  
  requestCount.add(1);
  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);
}

function testGraphQLAPI() {
  const query = `
    query GetModels {
      models {
        id
        name
        provider
        status
      }
    }
  `;
  
  const payload = JSON.stringify({ query: query });
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  const response = http.post(`${BASE_URL}/graphql`, payload, params);
  
  check(response, {
    'graphql status is 200': (r) => r.status === 200,
    'graphql has data': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.data && data.data.models;
      } catch {
        return false;
      }
    },
  });
  
  requestCount.add(1);
  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);
}

function testWebSocket() {
  const url = `ws://localhost:8000/ws/chat`;
  
  const res = ws.connect(url, {}, function (socket) {
    socket.on('open', () => {
      console.log('WebSocket connected');
      
      // Send test message
      socket.send(JSON.stringify({
        message: 'Hello via WebSocket!',
        model: 'llama2-7b'
      }));
    });
    
    socket.on('message', (data) => {
      try {
        const message = JSON.parse(data);
        check(message, {
          'websocket message has response': (m) => m.message && m.message.length > 0,
        });
      } catch (e) {
        console.log('Failed to parse WebSocket message:', e);
      }
    });
    
    socket.on('close', () => {
      console.log('WebSocket disconnected');
    });
    
    socket.on('error', (e) => {
      console.log('WebSocket error:', e);
      errorRate.add(1);
    });
    
    // Keep connection open for a short time
    sleep(2);
    socket.close();
  });
  
  check(res, {
    'websocket connection successful': (r) => r && r.status === 101,
  });
  
  requestCount.add(1);
  if (!res || res.status !== 101) {
    errorRate.add(1);
  }
}

// Stress test for high load scenarios
export function stressTest() {
  const options = {
    stages: [
      { duration: '1m', target: 50 },   // Ramp up to 50 users
      { duration: '3m', target: 50 },   // Stay at 50 users
      { duration: '1m', target: 100 },  // Ramp up to 100 users
      { duration: '3m', target: 100 },  // Stay at 100 users
      { duration: '2m', target: 0 },    // Ramp down
    ],
    thresholds: {
      http_req_duration: ['p(95)<5000'], // Allow higher latency under stress
      http_req_failed: ['rate<0.2'],     // Allow higher error rate under stress
    },
  };
  
  // Run the same tests but with higher concurrency
  testOpenAIAPI();
  sleep(0.1); // Shorter sleep for stress testing
}

// Spike test for sudden load increases
export function spikeTest() {
  const options = {
    stages: [
      { duration: '1m', target: 10 },   // Normal load
      { duration: '30s', target: 100 }, // Sudden spike
      { duration: '2m', target: 100 },  // Sustained spike
      { duration: '30s', target: 10 },  // Back to normal
      { duration: '1m', target: 10 },   // Recovery
    ],
  };
  
  // Test system behavior under sudden load spikes
  testOpenAIAPI();
  testGraphQLAPI();
  sleep(0.5);
}

// Volume test for extended periods
export function volumeTest() {
  const options = {
    stages: [
      { duration: '5m', target: 25 },   // Ramp up
      { duration: '30m', target: 25 },  // Extended run
      { duration: '5m', target: 0 },    // Ramp down
    ],
  };
  
  // Test system stability over extended periods
  testHealthCheck();
  testOpenAIAPI();
  sleep(2);
}