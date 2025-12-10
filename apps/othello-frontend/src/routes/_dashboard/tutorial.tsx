import {createFileRoute, Link} from '@tanstack/react-router'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList, BreadcrumbPage,
  BreadcrumbSeparator
} from "@/components/ui/breadcrumb.tsx";
import {SidebarTrigger} from "@/components/ui/sidebar.tsx";
import {Separator} from "@/components/ui/separator.tsx";
import {Button} from "@/components/ui/button.tsx";
import {OthelloTutorial} from "@/components/product/othello-tutorial.tsx";

export const Route = createFileRoute('/_dashboard/tutorial')({
  component: RouteComponent,
})

function RouteComponent() {
  return <>
    <header
      className="flex h-16 shrink-0 items-center justify-between gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
      <div className="flex flex-1 items-center gap-2 px-4">
        <SidebarTrigger className="-ml-1"/>
        <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4"/>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem className="hidden md:block">
              <BreadcrumbLink asChild>
                <Link to="/">Othello</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator className="hidden md:block"/>
            <BreadcrumbItem>
              <BreadcrumbPage>Tutorial</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>
      <div className="px-4">
      </div>
      <div className="flex flex-1 justify-end items-center gap-2 px-4">
          <Button variant="outline" asChild>
            <Link to="/play">Complete tutorial</Link>
          </Button>
      </div>
    </header>
    <div className="flex p-4 items-center justify-center w-full h-full">
      <div className="flex flex-col container max-w-4xl gap-4">
        <OthelloTutorial />
      </div>
    </div>
  </>
}