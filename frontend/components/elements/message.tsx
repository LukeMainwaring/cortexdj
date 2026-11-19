import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

export function MessageContent({
  className,
  children,
  ...props
}: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "text-foreground flex flex-col gap-2 overflow-hidden rounded-lg px-4 py-3 text-sm",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
