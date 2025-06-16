import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Product {
  id: string;
  description?: string;
  price?: number;
  country?: string;
  score?: number; // Renamed from confidence to match backend
}

interface ResultCardProps {
  title: string;
  naturalLanguageResponse?: string;
  extractedText?: string;
  identifiedProduct?: string;
  cnnClassName?: string;
  products?: Product[];
}

export function ResultCard({
  title,
  naturalLanguageResponse,
  extractedText,
  identifiedProduct,
  cnnClassName,
  products
}: ResultCardProps) {
  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-gray-900">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {naturalLanguageResponse && (
          <div>
            <h4 className="font-medium text-gray-700 mb-2">AI Response:</h4>
            <p className="text-gray-600 leading-relaxed">{naturalLanguageResponse}</p>
          </div>
        )}
        
        {extractedText && (
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Extracted Text:</h4>
            <div className="bg-gray-50 p-3 rounded-md">
              <p className="text-gray-800 whitespace-pre-wrap">{extractedText}</p>
            </div>
          </div>
        )}

        {identifiedProduct && (
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Identified Product:</h4>
            <p className="text-gray-600">{identifiedProduct}</p>
          </div>
        )}

        {cnnClassName && (
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Classification:</h4>
            <Badge variant="secondary">{cnnClassName}</Badge>
          </div>
        )}

        {products && products.length > 0 && (
          <div>
            <h4 className="font-medium text-gray-700 mb-3">Matching Products:</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Stock Code</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Unit Price</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Country</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {products.map((product, index) => (
                    <tr key={index}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{product.id}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{product.description}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{product.price ? `$${product.price.toFixed(2)}` : 'N/A'}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{product.country || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}