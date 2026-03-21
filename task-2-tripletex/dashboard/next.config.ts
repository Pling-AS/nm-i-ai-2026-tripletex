import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV === 'development';

const nextConfig: NextConfig = {
  ...(isDev ? {} : { output: 'export' }),
  basePath: '/dashboard',
  trailingSlash: true,
  ...(isDev ? {
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: `http://localhost:${process.env.API_PORT || '9000'}/api/:path*`,
        },
      ];
    }
  } : {})
};

export default nextConfig;
