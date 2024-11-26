FROM node:16-alpine
WORKDIR /vue-root
COPY package*.json ./
COPY dist ./dist
CMD ["npx", "serve", "dist"]
