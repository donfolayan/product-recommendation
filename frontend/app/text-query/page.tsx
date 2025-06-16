"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { LoadingSpinner } from "@/components/loading-spinner";
import { ResultCard } from "@/components/result-card";
import { Search, MessageSquare } from "lucide-react";

export default function TextQueryPage() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!query.trim()) {
      setError("Please enter a search query");
      return;
    }

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
    
    try {
      setLoading(true);
      setError(null);
      setResponse(null);
      
      const res = await fetch(`${backendUrl}/api/product-recommendation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || 'API error occurred');
      }
      
      setResponse({
        natural_language_response: data.response, // Map backend's 'response' to frontend's 'natural_language_response'
        product_matches: data.products // Map backend's 'products' to frontend's 'product_matches'
      });
    } catch (err: any) {
      setError(err.message || 'Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !loading) {
      handleSubmit();
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center space-y-4">
        <div className="flex justify-center">
          <div className="p-3 bg-blue-100 rounded-full">
            <MessageSquare className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl font-bold text-gray-900">Text-Based Product Query</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Describe what you're looking for in natural language and get intelligent product recommendations 
          powered by our AI system.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Search className="w-5 h-5" />
            <span>Search Products</span>
          </CardTitle>
          <CardDescription>
            Enter your product query in plain English. For example: "I need wireless headphones for running" 
            or "Looking for a laptop for graphic design under $1500"
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="queryInput">What are you looking for?</Label>
            <Input
              id="queryInput"
              placeholder="Describe the product you need..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading}
              className="text-base"
            />
          </div>
          
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          <Button 
            id="querySubmitButton"
            onClick={handleSubmit} 
            disabled={loading || !query.trim()}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <>
                <LoadingSpinner size="sm" className="mr-2" />
                Getting Recommendations...
              </>
            ) : (
              <>
                <Search className="w-4 h-4 mr-2" />
                Get Recommendations
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <div id="resultsDisplay">
        {response && (
          <ResultCard
            title="Search Results"
            naturalLanguageResponse={response.natural_language_response}
            products={response.product_matches}
          />
        )}
      </div>
    </div>
  );
}