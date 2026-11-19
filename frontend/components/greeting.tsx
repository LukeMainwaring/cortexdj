import Image from "next/image";
import { cn } from "@/lib/utils";

const FADE_UP =
  "fade-in slide-in-from-bottom-2 animate-in fill-mode-both duration-500";

export const Greeting = () => (
  <div className="mx-auto mt-4 flex size-full max-w-3xl flex-col justify-center px-4 md:mt-16 md:px-8">
    <div className="text-muted-foreground flex flex-col items-center gap-2 text-center">
      <div className={cn(FADE_UP, "delay-300")}>
        <Image
          alt="CortexDJ"
          className="size-20 rounded-full"
          height={64}
          src="/images/cortexdj-logo.png"
          width={64}
        />
      </div>
      <h2
        className={cn(
          FADE_UP,
          "text-foreground mt-2 text-lg font-semibold delay-500"
        )}
      >
        CortexDJ
      </h2>
      <p className={cn(FADE_UP, "max-w-md text-sm delay-600")}>
        An AI chatbot scaffold. Ask me anything to get started.
      </p>
    </div>
  </div>
);
