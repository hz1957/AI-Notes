import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

// https://vitepress.dev/reference/site-config
export default withMermaid({
  title: "HZ's AI notes",
  description: "HZ's AI notes",
  base: '/AI-Notes/',
  srcExclude: ['**/references/**'],
  markdown: {
    math: true
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Training', link: '/training/' },
      { text: 'Inference', link: '/inference/' },
      { text: 'Agent', link: '/agent/' }
    ],

    sidebar: {
      '/overview': [],
      '/training/': [
        {
          text: 'AI Training',
          items: [
            { text: 'Overview', link: '/training/' },
            { text: 'LoRA / QLoRA', link: '/training/lora' }
          ]
        }
      ],
      '/agent/': [
        {
          text: 'AI Agents',
          items: [
            { text: 'Overview', link: '/agent/' },
            { text: 'Practical Lessons / Case Studies', link: '/agent/practical-lessons' }
          ]
        }
      ],
      '/inference/': [
        {
          text: 'LLM Inference Optimization',
          items: [
            { text: 'Generation Parameters', link: '/inference/generation-parameters' },
            { text: 'KV Cache Strategies', link: '/inference/kv-cache' },
            { text: 'Model Optimization', link: '/inference/model-optimization' },
            { text: 'Parallelism Strategies', link: '/inference/parallelism' },
            { text: 'Serving Techniques', link: '/inference/serving-techniques' },
            { text: 'Speculative Decoding', link: '/inference/speculative-decoding' },
            { text: 'SGLang Configuration', link: '/inference/sglang-guide' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ]
  }
})
