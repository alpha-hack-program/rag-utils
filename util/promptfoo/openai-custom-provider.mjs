const promptfoo = require('promptfoo').default;

module.exports = class OpenAIProvider {
  constructor(options) {
    this.providerId = options.id || 'openai-custom';
    this.config = options.config || {};
  }

  id() {
    return this.providerId;
  }

  async callApi(prompt, context, options) {
    const apiBaseUrl = this.config.apiBaseUrl || 'https://api.openai.com/v1';
    const model = this.config.model || 'gpt-4.1-mini';
    const apiKey = this.config.apiKey || process.env.OPENAI_API_KEY;

    const { data } = await promptfoo.cache.fetchWithCache(
      `${apiBaseUrl}/chat/completions`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey && { Authorization: `Bearer ${apiKey}` }),
        },
        body: JSON.stringify({
          model,
          messages: [{ role: 'user', content: prompt }],
          max_tokens: this.config.max_tokens || 1024,
          temperature: this.config.temperature || 0,
          stream: false,
        }),
      }
    );

    return {
      output: data.choices[0].message.content,
      tokenUsage: data.usage,
    };
  }
};
