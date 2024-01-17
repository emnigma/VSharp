using Algorithms.Common;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualBasic.CompilerServices;
using Newtonsoft.Json;
using VSharp.ML.GameServer;


static class Extensions
{
    public static List<List<T>> Transpose<T>(this List<List<T>> list)
    {
        return list
            //maps to list of elems and their indices in original list [(element, index)]
            .SelectMany(x => x.Select<T, Tuple<T, int>>((e, i) => Tuple.Create<T, int>(e, i)))
            //groups tuples by their second element, their index
            .GroupBy(x => x.Item2)
            //this line is unnecessary but GroupBy doesn't guarantee an ordering
            .OrderBy(x => x.Key)
            //remove indices
            .Select(x => x.Select<Tuple<T, int>, T>(y => y.Item1).ToList()).ToList();
    }
}

namespace InferenceProvider
{
    public class Inference
    {
        public class NativeInput
        {
            public NativeInput(
                List<List<uint>> gameVertex, List<List<uint>> stateVertex,
                List<List<uint>> gameVertexToGameVertex, List<List<uint>> gameVertexHistoryStateVertexIndex,
                List<uint> gameVertexHistoryStateVertexAttrs, List<List<uint>> gameVertexInStateVertex,
                List<List<uint>> stateVertexParentOfStateVertex, Dictionary<uint, uint> stateMap)
            {
                this.gameVertex = gameVertex;
                this.stateVertex = stateVertex;
                this.gameVertexToGameVertex = gameVertexToGameVertex;
                this.gameVertexHistoryStateVertexIndex = gameVertexHistoryStateVertexIndex;
                this.gameVertexHistoryStateVertexAttrs = gameVertexHistoryStateVertexAttrs;
                this.gameVertexInStateVertex = gameVertexInStateVertex;
                this.stateVertexParentOfStateVertex = stateVertexParentOfStateVertex;
                this.stateMap = stateMap;
            }

            public List<List<uint>> gameVertex;
            public List<List<uint>> stateVertex;
            public List<List<uint>> gameVertexToGameVertex;
            public List<List<uint>> gameVertexHistoryStateVertexIndex;
            public List<uint> gameVertexHistoryStateVertexAttrs;
            public List<List<uint>> gameVertexInStateVertex;
            public List<List<uint>> stateVertexParentOfStateVertex;
            public Dictionary<uint, uint> stateMap;
        }

        private static NativeInput ConvertToNativeInput(Messages.GameState input)
        {
            var vertexMap = new Dictionary<uint, uint>();
            var nodesVertex = new List<List<uint>>();
            var edgesIndexVV = new List<List<uint>>();
            var edgesAttrsVV = new List<List<int>>();
            var edgesTypesVV = new List<int>();

            var stateMap = new Dictionary<uint, uint>();
            var nodesState = new List<List<uint>>();

            var edgesIndexSS = new List<List<uint>>();
            var edgesIndexSVIn = new List<List<uint>>();
            var edgesIndexVSIn = new List<List<uint>>();

            var edgesIndexSVHistory = new List<List<uint>>();
            var edgesIndexVSHistory = new List<List<uint>>();
            var edgesAttrsSV = new List<uint>();
            var edgesAttrsVS = new List<uint>();

            uint mappedVertexIndex = 0;
            uint mappedStateIndex = 0;
            // vertex nodes
            foreach (var vertex in input.GraphVertices)
            {
                if (!vertexMap.ContainsKey(vertex.Id))
                {
                    vertexMap[vertex.Id] = mappedVertexIndex; // maintain order in tensors
                    mappedVertexIndex += 1;
                    nodesVertex.Add(new List<uint>
                    {
                        vertex.InCoverageZone ? 1u : 0u,
                        vertex.BasicBlockSize,
                        vertex.CoveredByTest ? 1u : 0u,
                        vertex.VisitedByState ? 1u : 0u,
                        vertex.TouchedByState ? 1u : 0u,
                    });
                }
            }

            // vertex -> vertex edges
            foreach (var edge in input.Map)
            {
                edgesIndexVV.Add(new List<uint> { vertexMap[edge.VertexFrom], vertexMap[edge.VertexTo] });
                edgesAttrsVV.Add(new List<int> { edge.Label.Token });
                edgesTypesVV.Add(edge.Label.Token);
            }

            // state nodes
            foreach (var state in input.States)
            {
                if (!stateMap.ContainsKey(state.Id))
                {
                    stateMap[state.Id] = mappedStateIndex;
                    nodesState.Add(
                        new List<uint>
                        {
                            state.Position,
                            Convert.ToUInt32(state.PredictedUsefulness),
                            state.PathConditionSize,
                            state.VisitedAgainVertices,
                            state.VisitedNotCoveredVerticesInZone,
                            state.VisitedNotCoveredVerticesOutOfZone,
                        });
                    // history edges: state -> vertex and back
                    foreach (var historyElem in state.History)
                    {
                        var vertexTo = vertexMap[historyElem.GraphVertexId];
                        edgesIndexSVHistory.Add(new List<uint> { mappedStateIndex, vertexTo });
                        edgesIndexVSHistory.Add(new List<uint> { vertexTo, mappedStateIndex });
                        edgesAttrsSV.Add(historyElem.NumOfVisits);
                        edgesAttrsVS.Add(historyElem.NumOfVisits);
                    }

                    mappedStateIndex += 1;
                }
            }

            // state and its children edges: state -> state
            foreach (var state in input.States)
            {
                foreach (var child in state.Children)
                {
                    if (stateMap.ContainsKey(child))
                    {
                        edgesIndexSS.Add(new List<uint> { stateMap[state.Id], stateMap[child] });
                    }
                }
            }
            // state position edges: vertex -> state and back

            foreach (var vertex in input.GraphVertices)
            {
                foreach (var stateId in vertex.States)
                {
                    edgesIndexSVIn.Add(new List<uint> { stateMap[stateId], vertexMap[vertex.Id] });
                    edgesIndexVSIn.Add(new List<uint> { vertexMap[vertex.Id], stateMap[stateId] });
                }
            }

            return new NativeInput(
                nodesVertex,
                nodesState,
                edgesIndexVV,
                edgesIndexVSHistory,
                edgesAttrsVS,
                edgesIndexVSIn,
                edgesIndexSS,
                stateMap);
        }

        private static OrtValue Create2DFloatTensor(List<List<uint>> buffer)
        {
            var shape0 = buffer.Count;
            var shape1 = buffer[0].Count;
            float[] sourceData = buffer.SelectMany(i => i).Select(i => (float)i).ToArray();
            long[] dimensions = { shape0, shape1 };
            return OrtValue.CreateTensorValueFromMemory(sourceData, dimensions);
        }

        private static OrtValue Create2DLongTensor(List<List<uint>> buffer)
            // prev repeat
        {
            var shape0 = buffer.Count;
            var shape1 = buffer[0].Count;
            long[] sourceData = buffer.SelectMany(i => i).Select(i => (long)i).ToArray();
            long[] dimensions = { shape0, shape1 };
            return OrtValue.CreateTensorValueFromMemory(sourceData, dimensions);
        }

        private static uint PredictState(float[] ranks, Dictionary<uint, uint> stateMap)
        {
            var reverseMap = stateMap.ToDictionary(x => x.Value, x => x.Key);
            var (_, index) = ranks.Select((n, i) => (n, i)).Max();

            return reverseMap[Convert.ToUInt32(index)];
        }

        public static uint Infer(NativeInput nativeInput, InferenceSession session)
        {
            var input = new Dictionary<string, OrtValue>
            {
                { "game_vertex", Create2DFloatTensor(nativeInput.gameVertex) },
                { "state_vertex", Create2DFloatTensor(nativeInput.stateVertex) },
                { "game_vertex to game_vertex", Create2DLongTensor(nativeInput.gameVertexToGameVertex.Transpose()) },
                {
                    "game_vertex history state_vertex index",
                    Create2DLongTensor(nativeInput.gameVertexHistoryStateVertexIndex.Transpose())
                },
                {
                    "game_vertex history state_vertex attrs",
                    Create2DLongTensor(nativeInput.gameVertexHistoryStateVertexAttrs
                        .Select(item => new List<uint> { item }).ToList())
                },
                {
                    "game_vertex in state_vertex", Create2DLongTensor(nativeInput.gameVertexInStateVertex.Transpose())
                },
                {
                    "state_vertex parent_of state_vertex",
                    Create2DLongTensor(nativeInput.stateVertexParentOfStateVertex.Transpose())
                }
            };

            using var runOptions = new RunOptions();
            var output = session.Run(runOptions, input, session.OutputNames);
            var outputData = output[0].GetTensorDataAsSpan<float>().ToArray();

            var ranks = outputData.ToArray();

            return PredictState(ranks, nativeInput.stateMap);
        }

        public static void Main(string[] args)
        {
            var fileContents = File.ReadAllText("/Users/emax/Data/VSharp/InferenceProvider/test.json");
            var deserializeGameState = JsonConvert.DeserializeObject<Messages.GameState>(fileContents);

            Console.WriteLine(deserializeGameState);

            var nativeInput = ConvertToNativeInput(deserializeGameState);

            string modelPath = "/Users/emax/Data/VSharp/InferenceProvider/test_model_TAGSageSimple_ll.onnx";

            using var session = new InferenceSession(modelPath);
            var inferredStateId = Infer(nativeInput, session);

            Console.WriteLine(inferredStateId);
        }
    }
}