FROM node:20-alpine

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

# Security: Run as non-root user
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001
RUN npm install -g pnpm

COPY package.json pnpm-lock.yaml ./
RUN pnpm install

COPY . .

RUN pnpm build
RUN chown -R nextjs:nodejs /app

USER nextjs

EXPOSE 3000

CMD ["pnpm", "start"]
