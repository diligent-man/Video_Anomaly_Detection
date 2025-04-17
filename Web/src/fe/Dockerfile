FROM node:18-alpine

RUN adduser -D nonroot
USER nonroot

WORKDIR /Web/src/fe
RUN chown -R nonroot:nonroot /Web/src/fe

COPY --chown=nonroot:nonroot package*.json ./
RUN npm install

COPY --chown=nonroot:nonroot . .

EXPOSE 6969
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
