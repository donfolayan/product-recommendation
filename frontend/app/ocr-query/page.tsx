"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { LoadingSpinner } from "@/components/loading-spinner";
import { ResultCard } from "@/components/result-card";
import { Camera, Upload, FileText } from "lucide-react";

export default function OCRQueryPage() {
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
      setError("Please select an image file");
      return;
    }

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      setLoading(true);
      setError(null);
      setResponse(null);
      
      const res = await fetch(`${backendUrl}/api/ocr-query`, {
        method: 'POST',
        body: formData
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || 'OCR processing failed');
      }
      
      setResponse({
        extracted_text: data.ocr_result.text,
        ocr_confidence: data.ocr_result.confidence,
        engine_used: data.ocr_result.engine_used,
        processed_results: {
          naturalLanguageResponse: data.recommendations.response,
          products: data.recommendations.products
        }
      });
    } catch (err: any) {
      setError(err.message || 'Failed to process image');
      setResponse({
        extracted_text: "",
        ocr_confidence: 0.0,
        engine_used: "none",
        processed_results: {
          naturalLanguageResponse: "An error occurred. Please try again.",
          products: []
        }
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center space-y-4">
        <div className="flex justify-center">
          <div className="p-3 bg-green-100 rounded-full">
            <FileText className="w-8 h-8 text-green-600" />
          </div>
        </div>
        <h1 className="text-3xl font-bold text-gray-900">OCR-Based Query</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Upload images containing text and our OCR technology will extract the text and process 
          it to find relevant product recommendations.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Camera className="w-5 h-5" />
            <span>Image Upload</span>
          </CardTitle>
          <CardDescription>
            Select an image file containing text (receipts, product descriptions, shopping lists, etc.)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="imageFileInput">Choose Image File</Label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-gray-400 transition-colors">
              <input
                id="imageFileInput"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <label 
                htmlFor="imageFileInput" 
                className="cursor-pointer flex flex-col items-center space-y-2"
              >
                <Upload className="w-8 h-8 text-gray-400" />
                <span className="text-sm text-gray-600">
                  {selectedFile ? selectedFile.name : "Click to upload an image"}
                </span>
                <span className="text-xs text-gray-500">
                  Supports JPG, PNG, GIF up to 10MB
                </span>
              </label>
            </div>
          </div>

          {imagePreview && (
            <div className="space-y-2">
              <Label>Preview</Label>
              <div className="border rounded-lg p-4 bg-gray-50">
                <img 
                  src={imagePreview} 
                  alt="Upload preview" 
                  className="max-w-full h-auto max-h-64 mx-auto rounded"
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
            id="ocrSubmitButton"
            onClick={handleSubmit} 
            disabled={loading || !selectedFile}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <>
                <LoadingSpinner size="sm" className="mr-2" />
                Processing Image...
              </>
            ) : (
              <>
                <FileText className="w-4 h-4 mr-2" />
                Process Image
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <div id="ocrResultsDisplay">
        {response && (
          <ResultCard
            title="OCR Results"
            extractedText={response.extracted_text}
            naturalLanguageResponse={response.processed_results.naturalLanguageResponse}
            products={response.processed_results.products}
          />
        )}
      </div>
    </div>
  );
}