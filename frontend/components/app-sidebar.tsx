"use client";

import { Plus } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { SidebarHistory } from "@/components/sidebar-history";
import { SidebarUserNav } from "@/components/sidebar-user-nav";
import { Button } from "@/components/ui/button";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  useSidebar,
} from "@/components/ui/sidebar";
import { useStartNewChat } from "@/hooks/use-start-new-chat";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";

export function AppSidebar() {
  const { setOpenMobile } = useSidebar();
  const startNewChat = useStartNewChat();

  return (
    <Sidebar className="group-data-[side=left]:border-r-0">
      <SidebarHeader>
        <SidebarMenu>
          <div className="flex flex-row items-center justify-between">
            <Link
              className="hover:bg-muted flex flex-row items-center gap-1.5 rounded-md px-2 py-1"
              href="/"
              onClick={() => {
                setOpenMobile(false);
              }}
            >
              <Image
                alt="CortexDJ logo"
                className="rounded-sm"
                height={28}
                src="/images/cortexdj-logo.png"
                width={28}
              />
              <span className="text-lg font-semibold">CortexDJ</span>
            </Link>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  className="h-8 p-1 md:h-fit md:p-2"
                  onClick={() => {
                    setOpenMobile(false);
                    startNewChat();
                  }}
                  type="button"
                  variant="ghost"
                >
                  <Plus />
                </Button>
              </TooltipTrigger>
              <TooltipContent align="end" className="hidden md:block">
                New Chat
              </TooltipContent>
            </Tooltip>
          </div>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarHistory />
      </SidebarContent>
      <SidebarFooter>
        <SidebarUserNav />
      </SidebarFooter>
    </Sidebar>
  );
}
