import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Search, Camera, Image, ArrowRight } from "lucide-react";

const features = [
  {
    icon: Search,
    title: "Text Query",
    description: "Describe what you're looking for in natural language and get intelligent product recommendations",
    href: "/text-query",
    color: "text-blue-600"
  },
  {
    icon: Camera,
    title: "OCR Query", 
    description: "Upload images with text and extract product information automatically using OCR technology",
    href: "/ocr-query",
    color: "text-green-600"
  },
  {
    icon: Image,
    title: "Image Detection",
    description: "Upload product images and get instant identification with matching recommendations",
    href: "/image-upload",
    color: "text-purple-600"
  }
];

export default function Home() {
  return (
    <div className="max-w-2xl mx-auto space-y-6 pt-12">
      {features.map((feature, index) => {
        const Icon = feature.icon;
        return (
          <Card key={index} className="group hover:shadow-lg transition-all duration-300 border-0 shadow-md">
            <CardHeader className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-gray-50 rounded-lg group-hover:bg-blue-50 transition-colors">
                  <Icon className={`w-6 h-6 ${feature.color}`} />
                </div>
                <CardTitle className="text-xl">{feature.title}</CardTitle>
              </div>
              <CardDescription className="text-gray-600 leading-relaxed">
                {feature.description}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link href={feature.href}>
                <Button className="w-full group-hover:bg-blue-600 transition-colors">
                  Get Started
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}