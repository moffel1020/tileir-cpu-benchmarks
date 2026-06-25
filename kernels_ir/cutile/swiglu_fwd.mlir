cuda_tile.module @kernels {
  entry @swiglu_fwd(%arg0: tile<ptr<f32>>, %arg1: tile<i32>, %arg2: tile<i32>, %arg3: tile<i32>, %arg4: tile<i32>, %arg5: tile<ptr<f32>>, %arg6: tile<i32>, %arg7: tile<i32>, %arg8: tile<i32>, %arg9: tile<i32>, %arg10: tile<ptr<f32>>, %arg11: tile<i32>, %arg12: tile<i32>, %arg13: tile<i32>, %arg14: tile<i32>) optimization_hints=<sm_89 = {}> {
    %0 = make_token : token
    %assume = assume bounded<0, ?>, %arg1 : tile<i32>
    %assume_0 = assume bounded<0, ?>, %arg2 : tile<i32>
    %tview = make_tensor_view %arg0, shape = [%assume, %assume_0], strides = [1024, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[1024,1]>
    %assume_1 = assume bounded<0, ?>, %arg6 : tile<i32>
    %assume_2 = assume bounded<0, ?>, %arg7 : tile<i32>
    %tview_3 = make_tensor_view %arg5, shape = [%assume_1, %assume_2], strides = [1024, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[1024,1]>
    %assume_4 = assume bounded<0, ?>, %arg11 : tile<i32>
    %assume_5 = assume bounded<0, ?>, %arg12 : tile<i32>
    %tview_6 = make_tensor_view %arg10, shape = [%assume_4, %assume_5], strides = [1024, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[1024,1]>
    %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
    %cst_0_i32 = constant <i32: 0> : tile<i32>
    %pview = make_partition_view %tview : partition_view<tile=(1x1024), tensor_view<?x?xf32, strides=[1024,1]>>
    %tile, %result_token = load_view_tko weak %pview[%blockId_x, %cst_0_i32] token = %0 : partition_view<tile=(1x1024), tensor_view<?x?xf32, strides=[1024,1]>>, tile<i32> -> tile<1x1024xf32>, token
    %cst_0_i32_7 = constant <i32: 0> : tile<i32>
    %pview_8 = make_partition_view %tview_3 : partition_view<tile=(1x1024), tensor_view<?x?xf32, strides=[1024,1]>>
    %tile_9, %result_token_10 = load_view_tko weak %pview_8[%blockId_x, %cst_0_i32_7] token = %0 : partition_view<tile=(1x1024), tensor_view<?x?xf32, strides=[1024,1]>>, tile<i32> -> tile<1x1024xf32>, token
    %cst_1_f32 = constant <f32: 1.000000e+00> : tile<f32>
    %1 = negf %tile : tile<1x1024xf32>
    %2 = exp %1 : tile<1x1024xf32>
    %reshape = reshape %cst_1_f32 : tile<f32> -> tile<1x1xf32>
    %bcast = broadcast %reshape : tile<1x1xf32> -> tile<1x1024xf32>
    %3 = addf %bcast, %2  flush_to_zero : tile<1x1024xf32>
    %cst_1_f32_11 = constant <f32: 1.000000e+00> : tile<f32>
    %reshape_12 = reshape %cst_1_f32_11 : tile<f32> -> tile<1x1xf32>
    %bcast_13 = broadcast %reshape_12 : tile<1x1xf32> -> tile<1x1024xf32>
    %4 = divf %bcast_13, %3 rounding<approx> flush_to_zero : tile<1x1024xf32>
    %5 = mulf %tile, %4  flush_to_zero : tile<1x1024xf32>
    %6 = mulf %5, %tile_9  : tile<1x1024xf32>
    %cst_0_i32_14 = constant <i32: 0> : tile<i32>
    %pview_15 = make_partition_view %tview_6 : partition_view<tile=(1x1024), tensor_view<?x?xf32, strides=[1024,1]>>
    %7 = store_view_tko weak %6, %pview_15[%blockId_x, %cst_0_i32_14] token = %0 : tile<1x1024xf32>, partition_view<tile=(1x1024), tensor_view<?x?xf32, strides=[1024,1]>>, tile<i32> -> token
    return
  }
}
