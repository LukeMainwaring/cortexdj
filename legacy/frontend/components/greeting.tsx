import { motion } from "motion/react";
import Image from "next/image";

export const Greeting = () => {
  return (
    <div
      className="mx-auto mt-4 flex size-full max-w-3xl flex-col justify-center px-4 md:mt-16 md:px-8"
      key="overview"
    >
      <div className="flex flex-col items-center gap-2 text-center text-muted-foreground">
        <motion.div
          animate={{ opacity: 1, y: 0 }}
          initial={{ opacity: 0, y: 10 }}
          transition={{ delay: 0.3 }}
        >
          <Image
            alt="CortexDJ"
            className="size-20 rounded-full"
            height={64}
            src="/images/cortexdj-logo.png"
            width={64}
          />
        </motion.div>
        <motion.h2
          animate={{ opacity: 1, y: 0 }}
          className="mt-2 text-lg font-semibold text-foreground"
          initial={{ opacity: 0, y: 10 }}
          transition={{ delay: 0.5 }}
        >
          CortexDJ
        </motion.h2>
        <motion.p
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md text-sm"
          initial={{ opacity: 0, y: 10 }}
          transition={{ delay: 0.6 }}
        >
          Ask me to analyze EEG sessions, explain brain states, or build Spotify
          playlists based on your brain-wave data.
        </motion.p>
      </div>
    </div>
  );
};
