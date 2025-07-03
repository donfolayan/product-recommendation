"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { LoadingSpinner } from "@/components/loading-spinner";
import { ResultCard } from "@/components/result-card";
import { Image as ImageIcon, Upload, Scan } from "lucide-react";

export default function ImageUploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setResponse(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Please select a product image");
      return;
    }

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
      setLoading(true);
      setError(null);
      setResponse(null);
      
      const res = await fetch(`${backendUrl}/api/v1/product-detections`, {
        method: 'POST',
        body: formData
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || 'Product identification failed');
      }
      
      setResponse({
        identified_product_description: data.data.detected_class,
        cnn_class_name: data.data.detected_class,
        matching_products: data.data.similar_products
        });;
    } catch (err: any) {
      setError(err.message || 'Failed to identify product');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center space-y-4">
        <div className="flex justify-center">
          <div className="p-3 bg-purple-100 rounded-full">
            <Scan className="w-8 h-8 text-purple-600" />
          </div>
        </div>
        <h1 className="text-3xl font-bold text-gray-900">Product Image Upload</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Upload product images and our advanced CNN model will identify the product and 
          provide similar recommendations with detailed classifications.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <ImageIcon className="w-5 h-5" />
            <span>Product Image Detection</span>
          </CardTitle>
          <CardDescription>
            Upload a clear image of the product you want to identify. Works best with well-lit, 
            centered product photos.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="productImageInput">Choose Product Image</Label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 hover:border-gray-400 transition-colors">
              <input
                id="productImageInput"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <label 
                htmlFor="productImageInput" 
                className="cursor-pointer flex flex-col items-center space-y-3"
              >
                <div className="p-4 bg-purple-50 rounded-full">
                  <Upload className="w-8 h-8 text-purple-600" />
                </div>
                <div className="text-center">
                  <span className="text-base font-medium text-gray-700">
                    {selectedFile ? selectedFile.name : "Upload Product Image"}
                  </span>
                  <p className="text-sm text-gray-500 mt-1">
                    Drag and drop or click to browse
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    Supports JPG, PNG, GIF up to 10MB
                  </p>
                </div>
              </label>
            </div>
          </div>

          {imagePreview && (
            <div className="space-y-2">
              <Label>Image Preview</Label>
              <div className="border rounded-lg p-4 bg-gray-50">
                <img 
                  src={imagePreview} 
                  alt="Product preview" 
                  className="max-w-full h-auto max-h-80 mx-auto rounded-lg shadow-sm"
                />
              </div>
            </div>
          )}
          
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          <Button 
            id="identifySubmitButton"
            onClick={handleSubmit} 
            disabled={loading || !selectedFile}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <>
                <LoadingSpinner size="sm" className="mr-2" />
                Identifying Product...
              </>
            ) : (
              <>
                <Scan className="w-4 h-4 mr-2" />
                Identify Product
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <div id="identificationResultsDisplay">
        {response && (
          <ResultCard
            title="Product Identification Results"
            identifiedProduct={response.identified_product_description}
            cnnClassName={response.cnn_class_name}
            products={response.matching_products}
          />
        )}
      </div>
    </div>
  );
}