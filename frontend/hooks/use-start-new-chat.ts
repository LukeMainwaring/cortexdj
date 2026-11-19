import { useRouter } from "next/navigation";
import { useCallback } from "react";

// The refresh makes the new-chat page mint a fresh thread id even when the
// router already shows "/".
export function useStartNewChat() {
  const router = useRouter();
  return useCallback(() => {
    router.push("/");
    router.refresh();
  }, [router]);
}
