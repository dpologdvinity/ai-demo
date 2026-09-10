import * as React from "react";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export interface ParameterControlProps {
  label: string;
  value: number | string | boolean;
  onChange: (value: number | string | boolean) => void;
  min?: number;
  max?: number;
  step?: number;
  type?: "slider" | "number" | "select" | "range" | "text" | "boolean" | "toggle";
  options?: Array<{ value: string | number; label: string }>;
  description?: string;
  disabled?: boolean;
  className?: string;
}

export function ParameterControl({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  type = "slider",
  options = [],
  description,
  disabled,
  className,
}: ParameterControlProps) {
  const handleSliderChange = (values: number[]) => {
    onChange(values[0]);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(e.target.value);
    if (!isNaN(newValue)) {
      onChange(newValue);
    }
  };

  const handleSelectChange = (newValue: string) => {
    onChange(newValue);
  };

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
          {label}
        </label>
        {type !== "select" && (
          <span className="text-sm text-muted-foreground">{value}</span>
        )}
      </div>

      {description && (
        <p className="text-xs text-muted-foreground">{description}</p>
      )}

      {type === "slider" && (
        <div className="flex items-center gap-4">
          <Slider
            value={[Number(value)]}
            onValueChange={handleSliderChange}
            min={min}
            max={max}
            step={step}
            className="flex-1"
          />
          <Input
            type="number"
            value={Number(value)}
            onChange={handleInputChange}
            min={min}
            max={max}
            step={step}
            className="w-20"
          />
        </div>
      )}

      {type === "number" && (
        <Input
          type="number"
          value={Number(value)}
          onChange={handleInputChange}
          min={min}
          max={max}
          step={step}
          className="w-full"
        />
      )}

      {type === "select" && (
        <Select value={String(value)} onValueChange={handleSelectChange}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select an option" />
          </SelectTrigger>
          <SelectContent>
            {options.map((option) => (
              <SelectItem key={option.value} value={String(option.value)}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}

      {type === "text" && (
        <Input
          type="text"
          value={String(value)}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="w-full"
        />
      )}

      {(type === "boolean" || type === "toggle") && (
        <Switch
          checked={Boolean(value)}
          onCheckedChange={(checked) => onChange(checked)}
          disabled={disabled}
        />
      )}
    </div>
  );
}
