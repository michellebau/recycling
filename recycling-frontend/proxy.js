const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();

app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET,HEAD,PUT,PATCH,POST,DELETE");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});

// Logging middleware
app.use((req, res, next) => {
  console.log(`Received request: ${req.method} ${req.url}`);
  next();
});

// Error handling
app.use((error, req, res, next) => {
  console.error('Proxy error:', error);
  res.status(500).send('There was an error with the proxy.');
});

// Proxy middleware should come after logging and error handling
app.use('/', createProxyMiddleware({
  target: 'http://localhost:5000',
  changeOrigin: true,
  onProxyReq: (proxyReq, req, res) => {
    console.log(`Sending request to Flask: ${req.method} ${req.url}`);
  },
  onProxyRes: (proxyRes, req, res) => {
    console.log(`Received response from Flask: ${proxyRes.statusCode}`);
  },
  onError: (err, req, res) => {
    console.error(`Error in proxy: ${err.message}`);
  },
}));

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Proxy server running on http://localhost:${PORT}`);
});
