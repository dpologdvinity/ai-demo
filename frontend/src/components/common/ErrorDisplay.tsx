import * as React from "react";
import { AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface ErrorDisplayProps {
  error?: Error | string;
  message?: string;
  title?: string;
  retry?: () => void;
  className?: string;
}

export function ErrorDisplay({
  error,
  message,
  title = "An error occurred",
  retry,
  className,
}: ErrorDisplayProps) {
  const errorMessage = message || (typeof error === "string" ? error : error?.message || "Unknown error");

  return (
    <Card className={cn("border-destructive", className)}>
      <CardHeader>
        <div className="flex items-center gap-2">
          <AlertCircle className="h-5 w-5 text-destructive" />
          <CardTitle className="text-destructive">{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <CardDescription className="text-destructive/90">
          {errorMessage}
        </CardDescription>
      </CardContent>
      {retry && (
        <CardFooter>
          <Button onClick={retry} variant="outline" size="sm">
            Try Again
          </Button>
        </CardFooter>
      )}
    </Card>
  );
}
