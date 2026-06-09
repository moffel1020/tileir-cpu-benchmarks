cuda_tile.module @kernels {
  entry @layernorm_fwd(%arg0: tile<ptr<f32>>, %arg1: tile<i32>, %arg2: tile<i32>, %arg3: tile<i32>, %arg4: tile<i32>, %arg5: tile<ptr<f32>>, %arg6: tile<i32>, %arg7: tile<i32>, %arg8: tile<ptr<f32>>, %arg9: tile<i32>, %arg10: tile<i32>, %arg11: tile<ptr<f32>>, %arg12: tile<i32>, %arg13: tile<i32>, %arg14: tile<i32>, %arg15: tile<i32>, %arg16: tile<ptr<f32>>, %arg17: tile<i32>, %arg18: tile<i32>, %arg19: tile<ptr<f32>>, %arg20: tile<i32>, %arg21: tile<i32>) optimization_hints=<sm_89 = {}> {
    %0 = make_token : token
    %assume = assume bounded<0, ?>, %arg1 : tile<i32>
    %assume_0 = assume bounded<0, ?>, %arg2 : tile<i32>
    %tview = make_tensor_view %arg0, shape = [%assume, %assume_0], strides = [4096, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[4096,1]>
    %assume_1 = assume bounded<0, ?>, %arg6 : tile<i32>
    %tview_2 = make_tensor_view %arg5, shape = [%assume_1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
    %assume_3 = assume bounded<0, ?>, %arg9 : tile<i32>
    %tview_4 = make_tensor_view %arg8, shape = [%assume_3], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
    %assume_5 = assume bounded<0, ?>, %arg12 : tile<i32>
    %assume_6 = assume bounded<0, ?>, %arg13 : tile<i32>
    %tview_7 = make_tensor_view %arg11, shape = [%assume_5, %assume_6], strides = [4096, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[4096,1]>
    %assume_8 = assume bounded<0, ?>, %arg17 : tile<i32>
    %tview_9 = make_tensor_view %arg16, shape = [%assume_8], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
    %assume_10 = assume bounded<0, ?>, %arg20 : tile<i32>
    %tview_11 = make_tensor_view %arg19, shape = [%assume_10], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
    %cst_f32 = constant <f32: 9.99999974E-6> : tile<f32>
    %cst_16_i32 = constant <i32: 16> : tile<i32>
    %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
    %pview = make_partition_view %tview : partition_view<tile=(1x16), tensor_view<?x?xf32, strides=[4096,1]>>
    %1:2 = get_index_space_shape %pview : partition_view<tile=(1x16), tensor_view<?x?xf32, strides=[4096,1]>> -> tile<i32>
    %cst_0_f32 = constant <f32: 0.000000e+00> : tile<1x16xf32>
    %cst_0_i32 = constant <i32: 0> : tile<i32>
    %cst_1_i32 = constant <i32: 1> : tile<i32>
    %for = for %loopIdx in (%cst_0_i32 to %1#1, step %cst_1_i32) : tile<i32> iter_values(%iterArg0 = %cst_0_f32) -> (tile<1x16xf32>) {
      %pview_36 = make_partition_view %tview : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x?xf32, strides=[4096,1]>>
      %tile, %result_token = load_view_tko weak %pview_36[%blockId_x, %loopIdx] token = %0 : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x?xf32, strides=[4096,1]>>, tile<i32> -> tile<1x16xf32>, token
      %12 = addf %iterArg0, %tile  : tile<1x16xf32>
      continue %12 : tile<1x16xf32>
    }
    %reduce = reduce %for dim=1 identities=[0.000000e+00 : f32] : tile<1x16xf32> -> tile<1xf32> 
    (%reduce_lhs: tile<f32>, %reduce_rhs: tile<f32>) {
      %12 = addf %reduce_lhs, %reduce_rhs  : tile<f32>
      yield %12 : tile<f32>
    }
    %2 = itof %assume_0 signed  : tile<i32> -> tile<f32>
    %reshape = reshape %2 : tile<f32> -> tile<1xf32>
    %3 = divf %reduce, %reshape  : tile<1xf32>
    %pview_12 = make_partition_view %tview_9 : partition_view<tile=(1), tensor_view<?xf32, strides=[1]>>
    %4 = store_view_tko weak %3, %pview_12[%blockId_x] token = %0 : tile<1xf32>, partition_view<tile=(1), tensor_view<?xf32, strides=[1]>>, tile<i32> -> token
    %cst_0_f32_13 = constant <f32: 0.000000e+00> : tile<1x16xf32>
    %cst_0_i32_14 = constant <i32: 0> : tile<i32>
    %cst_1_i32_15 = constant <i32: 1> : tile<i32>
    %5 = iota : tile<16xi32>
    %reshape_16 = reshape %assume_0 : tile<i32> -> tile<1xi32>
    %bcast = broadcast %reshape_16 : tile<1xi32> -> tile<16xi32>
    %reshape_17 = reshape %3 : tile<1xf32> -> tile<1x1xf32>
    %bcast_18 = broadcast %reshape_17 : tile<1x1xf32> -> tile<1x16xf32>
    %cst_0_f32_19 = constant <f32: 0.000000e+00> : tile<f32>
    %reshape_20 = reshape %cst_0_f32_19 : tile<f32> -> tile<1x1xf32>
    %bcast_21 = broadcast %reshape_20 : tile<1x1xf32> -> tile<1x16xf32>
    %cst_2_f32 = constant <f32: 2.000000e+00> : tile<f32>
    %reshape_22 = reshape %cst_2_f32 : tile<f32> -> tile<1x1xf32>
    %bcast_23 = broadcast %reshape_22 : tile<1x1xf32> -> tile<1x16xf32>
    %for_24 = for %loopIdx in (%cst_0_i32_14 to %1#1, step %cst_1_i32_15) : tile<i32> iter_values(%iterArg0 = %cst_0_f32_13) -> (tile<1x16xf32>) {
      %pview_36 = make_partition_view %tview : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x?xf32, strides=[4096,1]>>
      %tile, %result_token = load_view_tko weak %pview_36[%blockId_x, %loopIdx] token = %0 : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x?xf32, strides=[4096,1]>>, tile<i32> -> tile<1x16xf32>, token
      %12 = muli %loopIdx, %cst_16_i32 : tile<i32>
      %reshape_37 = reshape %12 : tile<i32> -> tile<1xi32>
      %bcast_38 = broadcast %reshape_37 : tile<1xi32> -> tile<16xi32>
      %13 = addi %bcast_38, %5 : tile<16xi32>
      %14 = cmpi less_than %13, %bcast, signed : tile<16xi32> -> tile<16xi1>
      %15 = subf %tile, %bcast_18  : tile<1x16xf32>
      %reshape_39 = reshape %14 : tile<16xi1> -> tile<1x16xi1>
      %16 = select %reshape_39, %15, %bcast_21 : tile<1x16xi1>, tile<1x16xf32>
      %17 = pow %16, %bcast_23 : tile<1x16xf32>
      %18 = addf %iterArg0, %17  : tile<1x16xf32>
      continue %18 : tile<1x16xf32>
    }
    %reduce_25 = reduce %for_24 dim=1 identities=[0.000000e+00 : f32] : tile<1x16xf32> -> tile<1xf32> 
    (%reduce_lhs: tile<f32>, %reduce_rhs: tile<f32>) {
      %12 = addf %reduce_lhs, %reduce_rhs  : tile<f32>
      yield %12 : tile<f32>
    }
    %6 = itof %assume_0 signed  : tile<i32> -> tile<f32>
    %reshape_26 = reshape %6 : tile<f32> -> tile<1xf32>
    %7 = divf %reduce_25, %reshape_26  : tile<1xf32>
    %reshape_27 = reshape %cst_f32 : tile<f32> -> tile<1xf32>
    %8 = addf %7, %reshape_27  : tile<1xf32>
    %9 = sqrt %8  : tile<1xf32>
    %cst_1_f32 = constant <f32: 1.000000e+00> : tile<f32>
    %reshape_28 = reshape %cst_1_f32 : tile<f32> -> tile<1xf32>
    %10 = divf %reshape_28, %9  : tile<1xf32>
    %pview_29 = make_partition_view %tview_11 : partition_view<tile=(1), tensor_view<?xf32, strides=[1]>>
    %11 = store_view_tko weak %10, %pview_29[%blockId_x] token = %0 : tile<1xf32>, partition_view<tile=(1), tensor_view<?xf32, strides=[1]>>, tile<i32> -> token
    %cst_0_i32_30 = constant <i32: 0> : tile<i32>
    %cst_1_i32_31 = constant <i32: 1> : tile<i32>
    %reshape_32 = reshape %3 : tile<1xf32> -> tile<1x1xf32>
    %bcast_33 = broadcast %reshape_32 : tile<1x1xf32> -> tile<1x16xf32>
    %reshape_34 = reshape %10 : tile<1xf32> -> tile<1x1xf32>
    %bcast_35 = broadcast %reshape_34 : tile<1x1xf32> -> tile<1x16xf32>
    for %loopIdx in (%cst_0_i32_30 to %1#1, step %cst_1_i32_31) : tile<i32> {
      %pview_36 = make_partition_view %tview : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x?xf32, strides=[4096,1]>>
      %tile, %result_token = load_view_tko weak %pview_36[%blockId_x, %loopIdx] token = %0 : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x?xf32, strides=[4096,1]>>, tile<i32> -> tile<1x16xf32>, token
      %pview_37 = make_partition_view %tview_2 : partition_view<tile=(16), padding_value = zero, tensor_view<?xf32, strides=[1]>>
      %tile_38, %result_token_39 = load_view_tko weak %pview_37[%loopIdx] token = %0 : partition_view<tile=(16), padding_value = zero, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<16xf32>, token
      %pview_40 = make_partition_view %tview_4 : partition_view<tile=(16), padding_value = zero, tensor_view<?xf32, strides=[1]>>
      %tile_41, %result_token_42 = load_view_tko weak %pview_40[%loopIdx] token = %0 : partition_view<tile=(16), padding_value = zero, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<16xf32>, token
      %12 = subf %tile, %bcast_33  : tile<1x16xf32>
      %13 = mulf %12, %bcast_35  : tile<1x16xf32>
      %reshape_43 = reshape %tile_38 : tile<16xf32> -> tile<1x16xf32>
      %reshape_44 = reshape %tile_41 : tile<16xf32> -> tile<1x16xf32>
      %14 = fma %13, %reshape_43, %reshape_44  : tile<1x16xf32>
      %pview_45 = make_partition_view %tview_7 : partition_view<tile=(1x16), tensor_view<?x?xf32, strides=[4096,1]>>
      %15 = store_view_tko weak %14, %pview_45[%blockId_x, %loopIdx] token = %0 : tile<1x16xf32>, partition_view<tile=(1x16), tensor_view<?x?xf32, strides=[4096,1]>>, tile<i32> -> token
    }
    return
  }
}
