import { defineConfig } from 'vite'
import path from 'path'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  define: {
    // Only expose specific environment variables for security
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
  },
  server: {
    host: process.env.VITE_HOST || '0.0.0.0',
    port: parseInt(process.env.VITE_PORT || '3000'),
    strictPort: false,
    watch: {
      usePolling: true,
    },
    hmr: {
      clientPort: parseInt(process.env.VITE_PORT || '3000'),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  resolve: {
    alias: {
      // Use absolute path based on current directory so it works in Docker too
      '@': path.resolve(__dirname, 'src'),
    },
  },
  envPrefix: 'VITE_',
})