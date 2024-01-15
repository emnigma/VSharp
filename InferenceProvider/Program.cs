// See https://aka.ms/new-console-template for more information


using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics.Tensors;

namespace InferenceProvider
{
    public static class InferenceProvider
    {
        private static ReadOnlySpan<float> Run()
        {
            // Get path to model to create inference session.
            const string modelPath = "/Users/emax/Data/VSharp/InferenceProvider/test_nn.onnx";

            // Create an InferenceSession from the Model Path.
            // Creating and loading sessions are expensive per request.
            // They better be cached
            using var session = new InferenceSession(modelPath);

            float[] sourceData = { 1 }; // assume your data is loaded into a flat float array
            long[] dimensions = { 1 }; // and the dimensions of the input is stored here

            // Create a OrtValue on top of the sourceData array
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(sourceData, dimensions);

            var inputs = new Dictionary<string, OrtValue>
            {
                { "input", inputOrtValue }
            };

            using var runOptions = new RunOptions();

            // Pass inputs and request the first output
            // Note that the output is a disposable collection that holds OrtValues
            var output = session.Run(runOptions, inputs, session.OutputNames);

            var output_0 = output[0];

            // Assuming the output contains a tensor of float data, you can access it as follows
            // Returns Span<float> which points directly to native memory.
            var outputData = output_0.GetTensorDataAsSpan<float>();

            // If you are interested in more information about output, request its type and shape
            // Assuming it is a tensor
            // This is not disposable, will be GCed
            // There you can request Shape, ElementDataType, etc
            var tensorTypeAndShape = output_0.GetTensorTypeAndShape();
            
            Console.WriteLine(tensorTypeAndShape.ElementDataType);
            Console.WriteLine(output_0.GetTensorDataAsSpan<float>().ToArray());
            Console.WriteLine(output_0.Value);

            return outputData;
        }

        public static int Main(string[] args)
        {
            // Console.WriteLine("Hello World!");
            var outputs = Run();
            Console.WriteLine(outputs.ToArray());

            return 0;
        }
    }
}