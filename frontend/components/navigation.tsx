"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Search, Camera, Image, Zap } from "lucide-react";

const navigationItems = [
  {
    name: "Text Query",
    href: "/text-query",
    icon: Search,
    description: "Search with natural language"
  },
  {
    name: "OCR Query", 
    href: "/ocr-query",
    icon: Camera,
    description: "Extract text from images"
  },
  {
    name: "Image Upload",
    href: "/image-upload", 
    icon: Image,
    description: "Identify products from photos"
  }
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <Link href="/" className="flex items-center px-4 text-lg font-bold text-blue-600 hover:text-blue-700 transition-colors">
              <Zap className="w-6 h-6 mr-2" />
              ProductAI
            </Link>
          </div>
          <div className="flex space-x-8">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-all duration-200",
                    pathname === item.href
                      ? "border-blue-500 text-blue-600"
                      : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                  )}
                >
                  <Icon className="w-4 h-4 mr-2" />
                  {item.name}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}