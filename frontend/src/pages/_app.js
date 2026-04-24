import Head from "next/head";
import { CacheProvider } from "@emotion/react";
import CssBaseline from "@mui/material/CssBaseline";
import createEmotionCache from "@/theme/createEmotionCache";
import { ThemeModeProvider } from "@/context/ThemeModeContext";
import { AppSnackbarProvider } from "@/context/SnackbarContext";
import { AuthProvider } from "@/context/AuthContext";
import "@/styles/globals.css";

const clientSideEmotionCache = createEmotionCache();

export default function App(props) {
  const {
    Component,
    emotionCache = clientSideEmotionCache,
    pageProps
  } = props;
  const getLayout = Component.getLayout ?? ((page) => page);

  return (
    <CacheProvider value={emotionCache}>
      <Head>
        <title>FaceTrace Attendance</title>
        <meta
          content="Smart AI attendance system for colleges with live camera orchestration."
          name="description"
        />
        <meta content="initial-scale=1, width=device-width" name="viewport" />
      </Head>
      <ThemeModeProvider>
        <AppSnackbarProvider>
          <AuthProvider>
            <CssBaseline />
            {getLayout(<Component {...pageProps} />)}
          </AuthProvider>
        </AppSnackbarProvider>
      </ThemeModeProvider>
    </CacheProvider>
  );
}
