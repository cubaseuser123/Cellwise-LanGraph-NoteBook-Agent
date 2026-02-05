# Vercel AI SDK v6 Reference (Web Research)

## Core Concepts
The Vercel AI SDK v6 is split into **Core** (`ai`) and **UI** (`@ai-sdk/react`, etc.).

### Structured Output (`generateObject`)
For generating JSON matching a specific schema, `generateObject` is the standard function in the Core API.

```typescript
import { generateObject } from 'ai';
import { z } from 'zod';
import { google } from '@ai-sdk/google';

const result = await generateObject({
  model: google('gemini-1.5-flash'),
  schema: z.object({
    recipe: z.object({
      name: z.string(),
      ingredients: z.array(z.string()),
    }),
  }),
  prompt: 'Generate a lasagna recipe.',
});

console.log(result.object.recipe.name);
```

### Client-Side Usage
While typically used on the server, SDK Core functions can run client-side if the provider allows it and the API key is available.
- **Experimental Hooks**: `experimental_useObject` (or `useObject`) allows streaming structure to the client from a server endpoint.
- **Direct Client Calls**: Calling `generateObject` directly in a browser component is less common due to key exposure but functionally possible for internal tools.

## Migration Notes
- Ensure `@ai-sdk/google` is installed for Gemini.
- Use `createGoogleGenerativeAI` if granular config is needed.
- `generateObject` replaces older `experimental_` functions or manual JSON parsing.

## Google Provider
```typescript
import { createGoogleGenerativeAI } from '@ai-sdk/google';

const google = createGoogleGenerativeAI({
  apiKey: process.env.REACT_APP_GOOGLE_API_KEY,
});
```

## Tool Calling (CRITICAL)

> **⚠️ IMPORTANT**: In AI SDK v6, tools use **`inputSchema`**, NOT `parameters`! Using `parameters` will cause TypeScript errors and tool execution failures.

### Basic Tool Definition
```typescript
import { tool, jsonSchema } from 'ai';
import { z } from 'zod';

// Option 1: Using Zod (preferred for type safety)
const myTool = tool({
  description: 'Description of what the tool does',
  inputSchema: z.object({
    query: z.string().describe('The search query'),
  }),
  execute: async ({ query }) => {
    return `Result for: ${query}`;
  },
});

// Option 2: Using jsonSchema (needed for Gemini compatibility with Zod v4)
const myTool2 = tool({
  description: 'Search the knowledge base',
  inputSchema: jsonSchema<{ query: string }>({
    type: 'object',
    properties: {
      query: { type: 'string', description: 'The search query' }
    },
    required: ['query']
  }),
  execute: async (input: { query: string }) => {
    return `Result for: ${input.query}`;
  },
});
```

### Gemini + Zod v4 Compatibility Issue
When using **AI SDK v6 with Zod v4 and Google Gemini**, you may encounter:
```
functionDeclaration parameters schema should be of type OBJECT
```

**Solution**: Use `jsonSchema()` helper instead of Zod for tool schemas:
```typescript
import { jsonSchema } from 'ai';

// This works with Gemini + Zod v4
inputSchema: jsonSchema<{ query: string }>({
  type: 'object',
  properties: {
    query: { type: 'string' }
  },
  required: ['query']
})
```

### Using Tools with streamText
```typescript
import { streamText, tool, stepCountIs } from 'ai';

const result = streamText({
  model: gateway("google/gemini-2.5-flash"),
  tools: {
    search: tool({
      description: 'Search documents',
      inputSchema: z.object({ query: z.string() }),
      execute: async ({ query }) => searchDB(query),
    }),
  },
  stopWhen: stepCountIs(3), // Limit multi-step tool calls
  prompt: 'Search for...',
});
```

### Passing Context to Tools (experimental_context)
> **⚠️ CRITICAL**: Use `experimental_context`, NOT `toolContext`!

```typescript
const result = streamText({
  model: gateway("google/gemini-2.0-flash"),
  experimental_context: { userId: session.user.id }, // Pass context here
  tools: {
    myTool: tool({
      description: 'A tool that needs user context',
      inputSchema: jsonSchema<{ query: string }>({
        type: 'object',
        properties: { query: { type: 'string' } },
        required: ['query']
      }),
      // Context is received as second argument
      execute: async (input, { experimental_context }) => {
        const { userId } = experimental_context as { userId: string };
        return `User ${userId} searched for: ${input.query}`;
      },
    }),
  },
});
```

### Vercel AI Gateway
```typescript
import { gateway } from '@ai-sdk/gateway';

// Use provider/model format
model: gateway("google/gemini-2.5-flash")
model: gateway("mistral/devstral-2")
model: gateway("anthropic/claude-sonnet-4.5")
```

## Python Usage (OpenAI SDK Compatibility)

Vercel AI Gateway provides an OpenAI-compatible API, allowing Python projects to use the standard `openai` package.

### Setup
```bash
pip install openai python-dotenv
```

### Environment Variables
```bash
# .env
VERCEL_AI_GATEWAY_URL=https://gateway.ai.vercel.sh/v1  # Your gateway endpoint
VERCEL_AI_GATEWAY_API_KEY=your_vercel_api_key          # Your Vercel API key (or provider key)
```

### Basic Usage with OpenAI SDK
```python
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.getenv("VERCEL_AI_GATEWAY_URL"),
    api_key=os.getenv("VERCEL_AI_GATEWAY_API_KEY"),
)

# Use provider/model format (same as TypeScript)
response = client.chat.completions.create(
    model="mistral/devstral-2",  # or "google/gemini-2.5-flash"
    messages=[
        {"role": "user", "content": "Explain this code: def foo(): pass"}
    ],
    temperature=0.7,
    max_tokens=1000,
)

print(response.choices[0].message.content)
```

### Async Usage (for LangGraph)
```python
from openai import AsyncOpenAI
import asyncio

async_client = AsyncOpenAI(
    base_url=os.getenv("VERCEL_AI_GATEWAY_URL"),
    api_key=os.getenv("VERCEL_AI_GATEWAY_API_KEY"),
)

async def generate_explanation(code: str) -> str:
    response = await async_client.chat.completions.create(
        model="mistral/devstral-2",
        messages=[
            {"role": "system", "content": "You are a code documentation expert."},
            {"role": "user", "content": f"Explain this code:\n```python\n{code}\n```"}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content
```

### LangChain Integration (Alternative)
If using LangChain with LangGraph, you can use `ChatOpenAI` with custom base URL:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url=os.getenv("VERCEL_AI_GATEWAY_URL"),
    api_key=os.getenv("VERCEL_AI_GATEWAY_API_KEY"),
    model="mistral/devstral-2",
    temperature=0.7,
)

response = llm.invoke("Explain this code...")
```

> **Note**: The `provider/model` format (e.g., `mistral/devstral-2`) is used to route through the gateway to the appropriate provider.

