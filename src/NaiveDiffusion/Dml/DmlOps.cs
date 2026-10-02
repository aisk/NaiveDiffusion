using Vortice.DirectML;

namespace NaiveDiffusion.Dml;

/// <summary>The operator surface, in the mold of DirectMLX: each function adds
/// one node to the expression's graph and returns the output expression, with
/// the output shape computed here so callers never spell one out.</summary>
public static class DmlOps
{
    private static DmlTensorDesc PackedLike(DmlExpression source, uint[] sizes) =>
        DmlTensorDesc.Packed(source.Desc.DataType, sizes);

    /// <summary>The elementwise operators want two operands of one shape
    /// and one width: a mismatch in either is a graph error worth a sentence
    /// here rather than DirectML's own at compile time.</summary>
    private static void RequireSameShape(DmlExpression a, DmlExpression b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException(
                $"elementwise operands differ in shape: {a.Desc} vs {b.Desc}");
        }
        if (a.Desc.DataType != b.Desc.DataType)
        {
            throw new ArgumentException(
                $"elementwise operands differ in precision: {a.Desc} vs {b.Desc}");
        }
    }

    // --- Elementwise -------------------------------------------------------

    private static DmlExpression Binary(DmlExpression a, DmlExpression b,
        Func<TensorDescription, TensorDescription, TensorDescription, OperatorDescription> describe)
    {
        RequireSameShape(a, b);
        var output = PackedLike(a, a.Shape);
        return a.Graph.Emit(describe(a.Desc.ToTensor(), b.Desc.ToTensor(), output.ToTensor()),
            new[] { a, b }, output);
    }

    public static DmlExpression Add(DmlExpression a, DmlExpression b) => Binary(a, b,
        (x, y, output) => new ElementWiseAddOperatorDescription
            { ATensor = x, BTensor = y, OutputTensor = output });

    public static DmlExpression Subtract(DmlExpression a, DmlExpression b) => Binary(a, b,
        (x, y, output) => new ElementWiseSubtractOperatorDescription
            { ATensor = x, BTensor = y, OutputTensor = output });

    public static DmlExpression Multiply(DmlExpression a, DmlExpression b) => Binary(a, b,
        (x, y, output) => new ElementWiseMultiplyOperatorDescription
            { ATensor = x, BTensor = y, OutputTensor = output });

    public static DmlExpression Divide(DmlExpression a, DmlExpression b) => Binary(a, b,
        (x, y, output) => new ElementWiseDivideOperatorDescription
            { ATensor = x, BTensor = y, OutputTensor = output });

    /// <summary>output = input * scale + bias, elementwise. Also the packing
    /// copy: a strided view in, a packed tensor out.</summary>
    public static DmlExpression Identity(DmlExpression input, float scale = 1.0f, float bias = 0.0f)
    {
        var output = PackedLike(input, input.Shape);
        return input.Graph.Emit(new ElementWiseIdentityOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            ScaleBias = scale == 1.0f && bias == 0.0f ? null : new ScaleBias(scale, bias),
        }, new[] { input }, output);
    }

    /// <summary>output = (input * scale + bias) ^ exponent, elementwise.</summary>
    public static DmlExpression ConstantPow(DmlExpression input, float exponent,
        float scale = 1.0f, float bias = 0.0f)
    {
        var output = PackedLike(input, input.Shape);
        return input.Graph.Emit(new ElementWiseConstantPowOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            ScaleBias = scale == 1.0f && bias == 0.0f ? null : new ScaleBias(scale, bias),
            Exponent = exponent,
        }, new[] { input }, output);
    }

    /// <summary>The same values at another float width.</summary>
    public static DmlExpression Cast(DmlExpression input, TensorDataType dataType)
    {
        if (input.Desc.DataType == dataType)
        {
            return input;
        }
        var output = DmlTensorDesc.Packed(dataType, input.Shape);
        return input.Graph.Emit(new CastOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
        }, new[] { input }, output);
    }

    /// <summary>A reduction over <paramref name="axes"/>, which come out with
    /// extent 1 — the sum of squares along the last axis of a token tensor,
    /// for one.</summary>
    public static DmlExpression Reduce(DmlExpression input, int[] axes, ReduceFunction function)
    {
        uint[] outputSizes = input.Shape.ToArray();
        foreach (int axis in axes)
        {
            outputSizes[axis] = 1;
        }
        var output = PackedLike(input, outputSizes);
        return input.Graph.Emit(new ReduceOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            Function = function,
            Axes = axes,
        }, new[] { input }, output);
    }

    // --- Activations -------------------------------------------------------

    public static DmlExpression ActivationSigmoid(DmlExpression input)
    {
        var output = PackedLike(input, input.Shape);
        return input.Graph.Emit(new ActivationSigmoidOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
        }, new[] { input }, output);
    }

    /// <summary>Swish, input * sigmoid(input): SiLU as one operator, so the
    /// sigmoid never lands in memory. DML_FEATURE_LEVEL_6_2.</summary>
    public static DmlExpression ActivationSwish(DmlExpression input)
    {
        var output = PackedLike(input, input.Shape);
        return input.Graph.Emit(new ActivationSwishOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            SigmoidInputScale = 1.0f,
        }, new[] { input }, output);
    }

    /// <summary>The exact GELU, input * 0.5 * (1 + erf(input / sqrt(2))).</summary>
    public static DmlExpression ActivationGelu(DmlExpression input)
    {
        var output = PackedLike(input, input.Shape);
        return input.Graph.Emit(new ActivationGeluOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
        }, new[] { input }, output);
    }

    public static DmlExpression ActivationTanh(DmlExpression input)
    {
        var output = PackedLike(input, input.Shape);
        return input.Graph.Emit(new ActivationTanhOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
        }, new[] { input }, output);
    }

    public static DmlExpression ActivationIdentity(DmlExpression input)
    {
        var output = PackedLike(input, input.Shape);
        return input.Graph.Emit(new ActivationIdentityOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
        }, new[] { input }, output);
    }

    public static DmlExpression ActivationSoftmax(DmlExpression input, int[] axes)
    {
        var output = PackedLike(input, input.Shape);
        return input.Graph.Emit(new ActivationSoftmax1OperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            Axes = axes,
        }, new[] { input }, output);
    }

    // --- Views -------------------------------------------------------------

    /// <summary>The same bytes through different sizes and strides — no data
    /// moves. A stride of 0 repeats an axis, which is what broadcasting is.</summary>
    public static DmlExpression Reinterpret(DmlExpression input, uint[] sizes, uint[]? strides = null)
    {
        return input.WithDesc(input.Desc.Reinterpret(sizes, strides));
    }

    public static DmlExpression Reinterpret(DmlExpression input, int[] sizes, int[]? strides = null)
    {
        return Reinterpret(input,
            sizes.Select(extent => checked((uint)extent)).ToArray(),
            strides?.Select(stride => checked((uint)stride)).ToArray());
    }

    // --- Linear algebra ----------------------------------------------------

    /// <summary>Block dequantization: output = input * scale, where every axis
    /// of <paramref name="scale"/> divides the same axis of the input and each
    /// scale covers the block of elements it spans. The output takes the scale's
    /// float type. DML_FEATURE_LEVEL_6_3.</summary>
    public static DmlExpression Dequantize(DmlExpression input, DmlExpression scale)
    {
        uint[] inputSizes = input.Shape;
        uint[] scaleSizes = scale.Shape;
        if (inputSizes.Length != scaleSizes.Length ||
            inputSizes.Zip(scaleSizes).Any(pair => pair.First % pair.Second != 0))
        {
            throw new ArgumentException(
                $"scale {scale.Desc} does not tile the quantized tensor {input.Desc}");
        }

        var output = DmlTensorDesc.Packed(scale.Desc.DataType, inputSizes);
        return input.Graph.Emit(new DequantizeOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            QuantizationType = QuantizationType.Scale,
            QuantizationTensors = new[] { scale.Desc.ToTensor() },
            OutputTensor = output.ToTensor(),
        }, new[] { input, scale }, output);
    }

    /// <summary>output = alpha * transA(a) x transB(b) + beta * c, batched over
    /// the two leading axes.</summary>
    public static DmlExpression Gemm(DmlExpression a, DmlExpression b, DmlExpression? c = null,
        MatrixTransform transA = MatrixTransform.None, MatrixTransform transB = MatrixTransform.None,
        float alpha = 1.0f, float beta = 1.0f)
    {
        uint[] aShape = a.Shape;
        uint[] bShape = b.Shape;
        uint rows = transA == MatrixTransform.Transpose ? aShape[^1] : aShape[^2];
        uint inner = transA == MatrixTransform.Transpose ? aShape[^2] : aShape[^1];
        uint innerB = transB == MatrixTransform.Transpose ? bShape[^1] : bShape[^2];
        uint columns = transB == MatrixTransform.Transpose ? bShape[^2] : bShape[^1];
        if (inner != innerB)
        {
            throw new ArgumentException($"gemm inner extents differ: {a.Desc} vs {b.Desc}");
        }

        uint[] outputSizes = aShape.ToArray();
        outputSizes[^2] = rows;
        outputSizes[^1] = columns;

        var output = PackedLike(a, outputSizes);
        return a.Graph.Emit(new GeneralMatrixMultiplyOperatorDescription
        {
            ATensor = a.Desc.ToTensor(),
            BTensor = b.Desc.ToTensor(),
            CTensor = c?.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            TransformA = transA,
            TransformB = transB,
            Alpha = alpha,
            Beta = beta,
        }, new[] { a, b, c }, output);
    }

    /// <summary>A 2-D convolution (cross-correlation, as every framework means it).</summary>
    public static DmlExpression Convolution(DmlExpression input, DmlExpression filter,
        DmlExpression? bias, int[] strides, int[] startPadding, int[] endPadding,
        int[]? dilations = null, int groupCount = 1)
    {
        dilations ??= new[] { 1, 1 };
        uint[] inputSizes = input.Shape;
        uint[] filterSizes = filter.Shape;

        var outputSizes = new uint[inputSizes.Length];
        outputSizes[0] = inputSizes[0];
        outputSizes[1] = filterSizes[0];
        for (int d = 0; d < strides.Length; d++)
        {
            long padded = inputSizes[2 + d] + startPadding[d] + endPadding[d];
            long kernel = (filterSizes[2 + d] - 1L) * dilations[d] + 1;
            outputSizes[2 + d] = checked((uint)((padded - kernel) / strides[d] + 1));
        }

        var output = PackedLike(input, outputSizes);
        return input.Graph.Emit(new ConvolutionOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            FilterTensor = filter.Desc.ToTensor(),
            BiasTensor = bias?.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            Mode = ConvolutionMode.CrossCorrelation,
            Direction = ConvolutionDirection.Forward,
            Strides = strides,
            Dilations = dilations,
            StartPadding = startPadding,
            EndPadding = endPadding,
            OutputPadding = new int[strides.Length],
            GroupCount = groupCount,
        }, new[] { input, filter, bias }, output);
    }

    // --- Normalization -----------------------------------------------------

    /// <summary>output = (input - mean) / sqrt(variance + epsilon), with the mean
    /// and variance computed over <paramref name="axes"/>.</summary>
    public static DmlExpression MeanVarianceNormalization(DmlExpression input, int[] axes,
        float epsilon)
    {
        var output = PackedLike(input, input.Shape);
        return input.Graph.Emit(new MeanVarianceNormalization1OperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            ScaleTensor = null,
            BiasTensor = null,
            OutputTensor = output.ToTensor(),
            Axes = axes,
            NormalizeVariance = true,
            Epsilon = epsilon,
        }, new DmlExpression?[] { input, null, null }, output);
    }

    /// <summary>output = (input - mean) / sqrt(variance + epsilon) * scale + bias
    /// over <paramref name="axes"/>, where <paramref name="useMean"/> false
    /// takes the variance about zero — RMS normalization. Unlike the first
    /// version, <paramref name="scale"/> and <paramref name="bias"/> may vary
    /// along the normalized axes; each is the input's rank, sizes matching or
    /// 1.</summary>
    public static DmlExpression MeanVarianceNormalization2(DmlExpression input, int[] axes,
        bool useMean, float epsilon, DmlExpression? scale = null, DmlExpression? bias = null)
    {
        var output = PackedLike(input, input.Shape);
        return input.Graph.Emit(new MeanVarianceNormalization2OperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            ScaleTensor = scale?.Desc.ToTensor(),
            BiasTensor = bias?.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            Axes = axes,
            UseMean = useMean,
            UseVariance = true,
            Epsilon = epsilon,
        }, new DmlExpression?[] { input, scale, bias }, output);
    }

    // --- Shape operators ---------------------------------------------------

    /// <summary>A window into the input: <paramref name="sizes"/> elements taken
    /// from <paramref name="offsets"/> at <paramref name="strides"/> steps.</summary>
    public static DmlExpression Slice(DmlExpression input, int[] offsets, int[] sizes, int[] strides)
    {
        var outputSizes = new uint[sizes.Length];
        for (int d = 0; d < sizes.Length; d++)
        {
            outputSizes[d] = checked((uint)((sizes[d] + Math.Abs(strides[d]) - 1) / Math.Abs(strides[d])));
        }
        var output = PackedLike(input, outputSizes);
        return input.Graph.Emit(new Slice1OperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            InputWindowOffsets = offsets,
            InputWindowSizes = sizes,
            InputWindowStrides = strides,
        }, new[] { input }, output);
    }

    public static DmlExpression Join(IReadOnlyList<DmlExpression> inputs, int axis)
    {
        uint[] outputSizes = inputs[0].Shape.ToArray();
        for (int i = 1; i < inputs.Count; i++)
        {
            outputSizes[axis] += inputs[i].Shape[axis];
        }
        var output = PackedLike(inputs[0], outputSizes);
        return inputs[0].Graph.Emit(new JoinOperatorDescription
        {
            InputTensors = inputs.Select(input => input.Desc.ToTensor()).ToArray(),
            OutputTensor = output.ToTensor(),
            Axis = axis,
        }, inputs.ToArray(), output);
    }

    /// <summary>Gathers along <paramref name="axis"/> using the trailing
    /// <paramref name="indexDimensions"/> axes of <paramref name="indices"/>.</summary>
    public static DmlExpression Gather(DmlExpression input, DmlExpression indices,
        int axis, int indexDimensions)
    {
        uint[] inputSizes = input.Shape;
        uint[] indexSizes = indices.Shape;
        int rank = inputSizes.Length;

        // DirectMLX's shape rule: dimensions after the axis copy the input,
        // the index dimensions replace the axis walking backwards, and
        // whatever remains in front collapses to 1.
        var outputSizes = new uint[rank];
        Array.Fill(outputSizes, 1u);
        int outputDim = rank - 1;
        for (; outputDim > axis; outputDim--)
        {
            outputSizes[outputDim] = inputSizes[outputDim];
        }
        int indexDim = rank - 1;
        for (; outputDim > axis - indexDimensions; outputDim--, indexDim--)
        {
            outputSizes[outputDim] = indexSizes[indexDim];
        }

        var output = PackedLike(input, outputSizes);
        return input.Graph.Emit(new GatherOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            IndicesTensor = indices.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            Axis = axis,
            IndexDimensions = indexDimensions,
        }, new[] { input, indices }, output);
    }

    public static DmlExpression Upsample2D(DmlExpression input, uint scaleWidth, uint scaleHeight,
        InterpolationMode interpolationMode)
    {
        uint[] inputSizes = input.Shape;
        uint[] outputSizes =
        {
            inputSizes[0], inputSizes[1],
            inputSizes[2] * scaleHeight, inputSizes[3] * scaleWidth,
        };
        var output = PackedLike(input, outputSizes);
        return input.Graph.Emit(new Upsample2DOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            ScaleSize = new Size2D { Width = scaleWidth, Height = scaleHeight },
            InterpolationMode = interpolationMode,
        }, new[] { input }, output);
    }

    /// <summary>Average pooling over [1, C, H, W] with a square window and
    /// the same stride, no padding: H and W must divide by <paramref name="window"/>.</summary>
    public static DmlExpression AveragePooling(DmlExpression input, int window)
    {
        uint[] inputSizes = input.Shape;
        if (inputSizes.Length != 4 || inputSizes[2] % window != 0 || inputSizes[3] % window != 0)
        {
            throw new ArgumentException($"{input.Desc} does not pool by {window}");
        }
        uint[] outputSizes =
        {
            inputSizes[0], inputSizes[1], inputSizes[2] / (uint)window, inputSizes[3] / (uint)window,
        };
        var output = PackedLike(input, outputSizes);
        return input.Graph.Emit(new AveragePoolingOperatorDescription
        {
            InputTensor = input.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            Strides = new[] { window, window },
            WindowSize = new[] { window, window },
            StartPadding = new[] { 0, 0 },
            EndPadding = new[] { 0, 0 },
            IncludePadding = false,
        }, new[] { input }, output);
    }

    // --- Attention ---------------------------------------------------------

    /// <summary>Scaled dot-product attention as one operator, so the score
    /// matrix never becomes a tensor the graph has to find room for.
    ///
    /// The operator wants [1, batch, tokens, width] and the graphs here carry
    /// [batch, 1, tokens, width]; those are the same bytes, so the batch axis
    /// is moved across by a view on the way in and back on the way out.
    ///
    /// With a <paramref name="mask"/> the scores it rules out get
    /// <paramref name="maskFilterValue"/> added before the softmax. DirectML
    /// wants both kinds as int32: a boolean mask as [batch, heads, tokens,
    /// keys], 1 where a query may read a key — a [1, 1, tokens, keys] table
    /// gets there through a zero stride over the heads — and a key-length
    /// mask as [1, 1, 1, 1] holding how many leading keys are real, for a
    /// key sequence padded at its end. Anything else, a 1-D count or a table
    /// with heads of 1, is E_INVALIDARG at creation (see the smoke test).</summary>
    public static DmlExpression MultiheadAttention(DmlExpression query, DmlExpression key,
        DmlExpression value, int headCount, float scale, DmlExpression? mask = null,
        MultiheadAttentionMaskType maskType = MultiheadAttentionMaskType.None,
        float maskFilterValue = -10000f)
    {
        if ((mask is null) != (maskType == MultiheadAttentionMaskType.None))
        {
            throw new ArgumentException("a mask and its type come together");
        }
        query = MoveBatchAxis(query, toSecond: true);
        key = MoveBatchAxis(key, toSecond: true);
        value = MoveBatchAxis(value, toSecond: true);

        uint[] outputSizes = query.Shape.ToArray();
        outputSizes[^1] = value.Shape[^1];

        var output = PackedLike(query, outputSizes);
        DmlExpression attended = query.Graph.Emit(new MultiheadAttentionOperatorDescription
        {
            QueryTensor = query.Desc.ToTensor(),
            KeyTensor = key.Desc.ToTensor(),
            ValueTensor = value.Desc.ToTensor(),
            MaskTensor = mask?.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            Scale = scale,
            // Left at its default of 0 without a mask, so an unmasked
            // attention's description does not depend on the argument.
            MaskFilterValue = mask is null ? 0f : maskFilterValue,
            HeadCount = headCount,
            MaskType = maskType,
        }, new DmlExpression?[]
        {
            query, key, value,
            null, null, null,   // stacked query-key, key-value, query-key-value
            null, mask,         // bias, mask
            null,               // relative position bias
            null, null,         // past key, past value
        }, output);

        return MoveBatchAxis(attended, toSecond: false);
    }

    /// <summary>Unmasked attention over one stacked tensor of
    /// [batch, tokens, heads, 3, headSize], the query, key and value of each
    /// head side by side — the layout drivers' fused attention kernels tend to
    /// want. The output is [batch, 1, tokens, heads * headSize] as for the
    /// separate form.</summary>
    public static DmlExpression MultiheadAttention(DmlExpression stackedQueryKeyValue,
        int headCount, float scale)
    {
        uint[] sizes = stackedQueryKeyValue.Shape;
        if (sizes.Length != 5 || sizes[2] != headCount || sizes[3] != 3)
        {
            throw new ArgumentException(
                $"{stackedQueryKeyValue.Desc} is not [batch, tokens, {headCount}, 3, headSize]");
        }
        var output = PackedLike(stackedQueryKeyValue,
            new[] { 1u, sizes[0], sizes[1], sizes[2] * sizes[4] });
        DmlExpression attended = stackedQueryKeyValue.Graph.Emit(new MultiheadAttentionOperatorDescription
        {
            StackedQueryKeyValueTensor = stackedQueryKeyValue.Desc.ToTensor(),
            OutputTensor = output.ToTensor(),
            Scale = scale,
            HeadCount = headCount,
            MaskType = MultiheadAttentionMaskType.None,
        }, new DmlExpression?[]
        {
            null, null, null,
            null, null, stackedQueryKeyValue,
            null, null,
            null,
            null, null,
        }, output);

        return MoveBatchAxis(attended, toSecond: false);
    }

    private static DmlExpression MoveBatchAxis(DmlExpression x, bool toSecond)
    {
        uint[] sizes = x.Shape;
        uint batch = sizes[toSecond ? 0 : 1];
        if (batch == 1)
        {
            return x;
        }
        uint[] moved = sizes.ToArray();
        moved[0] = toSecond ? 1 : batch;
        moved[1] = toSecond ? batch : 1;
        return Reinterpret(x, moved);
    }
}
