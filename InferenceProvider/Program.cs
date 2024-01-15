// See https://aka.ms/new-console-template for more information

using VSharp.ML.GameServer;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics.Tensors;


namespace MyNamespace
{
    public class MyClass
    {
        public void Main()
        {
            // Get path to model to create inference session.
            string modelPath = "/Users/emax/Data/VSharp/InferenceProvider/test_nn.onnx";

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

            // Assuming the output contains a tensor of float data, you can access it as follows
            // Returns Span<float> which points directly to native memory.
            var outputData = output[0].GetTensorDataAsSpan<float>();

            // If you are interested in more information about output, request its type and shape
            // Assuming it is a tensor
            // This is not disposable, will be GCed
            // There you can request Shape, ElementDataType, etc
            var tensorTypeAndShape = output[0].GetTensorTypeAndShape();

            Console.WriteLine(tensorTypeAndShape.ElementDataType);
            Console.WriteLine(outputData.Length);
            Console.WriteLine(outputData.ToArray());

            foreach (var outp in outputData)
            {
                Console.WriteLine(outp);
            }
        }
    }
}