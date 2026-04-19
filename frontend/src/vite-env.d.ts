/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
  readonly VITE_ELEVENLABS_API_KEY?: string;
  readonly VITE_ELEVENLABS_MODEL?: string;
  readonly VITE_ALEX_VOICE_ID?: string;
  readonly VITE_SAM_VOICE_ID?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
