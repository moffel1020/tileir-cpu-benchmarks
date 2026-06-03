cuda_tile.module @kernels {
  entry @layernorm_fwd(%arg0: tile<ptr<f32>>, %arg1: tile<i32>, %arg2: tile<i32>, %arg3: tile<i32>, %arg4: tile<i32>, %arg5: tile<ptr<f32>>, %arg6: tile<i32>, %arg7: tile<i32>, %arg8: tile<ptr<f32>>, %arg9: tile<i32>, %arg10: tile<i32>, %arg11: tile<ptr<f32>>, %arg12: tile<i32>, %arg13: tile<i32>, %arg14: tile<i32>, %arg15: tile<i32>, %arg16: tile<ptr<f32>>, %arg17: tile<i32>, %arg18: tile<i32>, %arg19: tile<ptr<f32>>, %arg20: tile<i32>, %arg21: tile<i32>, %arg22: tile<f32>) optimization_hints=<sm_89 = {}> {
    %0 = make_token : token
    %assume = assume bounded<0, ?>, %arg1 : tile<i32>
    %assume_0 = assume bounded<0, ?>, %arg2 : tile<i32>
    %assume_1 = assume bounded<0, ?>, %arg3 : tile<i32>
    %assume_2 = assume bounded<0, ?>, %arg4 : tile<i32>
    %tview = make_tensor_view %arg0, shape = [%assume, %assume_0], strides = [%assume_1, %assume_2] : tile<i32> -> tensor_view<?x?xf32, strides=[?,?]>
    %assume_3 = assume bounded<0, ?>, %arg6 : tile<i32>
    %assume_4 = assume bounded<0, ?>, %arg7 : tile<i32>
    %tview_5 = make_tensor_view %arg5, shape = [%assume_3], strides = [%assume_4] : tile<i32> -> tensor_view<?xf32, strides=[?]>
    %assume_6 = assume bounded<0, ?>, %arg9 : tile<i32>
    %assume_7 = assume bounded<0, ?>, %arg10 : tile<i32>
    %tview_8 = make_tensor_view %arg8, shape = [%assume_6], strides = [%assume_7] : tile<i32> -> tensor_view<?xf32, strides=[?]>
    %assume_9 = assume bounded<0, ?>, %arg12 : tile<i32>
    %assume_10 = assume bounded<0, ?>, %arg13 : tile<i32>
    %assume_11 = assume bounded<0, ?>, %arg14 : tile<i32>
    %assume_12 = assume bounded<0, ?>, %arg15 : tile<i32>
    %tview_13 = make_tensor_view %arg11, shape = [%assume_9, %assume_10], strides = [%assume_11, %assume_12] : tile<i32> -> tensor_view<?x?xf32, strides=[?,?]>
    %assume_14 = assume bounded<0, ?>, %arg17 : tile<i32>
    %assume_15 = assume bounded<0, ?>, %arg18 : tile<i32>
    %tview_16 = make_tensor_view %arg16, shape = [%assume_14], strides = [%assume_15] : tile<i32> -> tensor_view<?xf32, strides=[?]>
    %assume_17 = assume bounded<0, ?>, %arg20 : tile<i32>
    %assume_18 = assume bounded<0, ?>, %arg21 : tile<i32>
    %tview_19 = make_tensor_view %arg19, shape = [%assume_17], strides = [%assume_18] : tile<i32> -> tensor_view<?xf32, strides=[?]>
    %cst_4096_i32 = constant <i32: 4096> : tile<i32>
    %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
    %pview = make_partition_view %tview : partition_view<tile=(1x4096), tensor_view<?x?xf32, strides=[?,?]>>
    %1:2 = get_index_space_shape %pview : partition_view<tile=(1x4096), tensor_view<?x?xf32, strides=[?,?]>> -> tile<i32>
    %cst_0_f32 = constant <f32: 0.000000e+00> : tile<1x4096xf32>
    %cst_0_i32 = constant <i32: 0> : tile<i32>
    %cst_1_i32 = constant <i32: 1> : tile<i32>
    %for = for %loopIdx in (%cst_0_i32 to %1#1, step %cst_1_i32) : tile<i32> iter_values(%iterArg0 = %cst_0_f32) -> (tile<1x4096xf32>) {
      %pview_44 = make_partition_view %tview : partition_view<tile=(1x4096), padding_value = zero, tensor_view<?x?xf32, strides=[?,?]>>
      %tile, %result_token = load_view_tko weak %pview_44[%blockId_x, %loopIdx] token = %0 : partition_view<tile=(1x4096), padding_value = zero, tensor_view<?x?xf32, strides=[?,?]>>, tile<i32> -> tile<1x4096xf32>, token
      %12 = addf %iterArg0, %tile  : tile<1x4096xf32>
      continue %12 : tile<1x4096xf32>
    }
    %reduce = reduce %for dim=1 identities=[0.000000e+00 : f32] : tile<1x4096xf32> -> tile<1xf32> 
    (%reduce_lhs: tile<f32>, %reduce_rhs: tile<f32>) {
      %12 = addf %reduce_lhs, %reduce_rhs  : tile<f32>
      yield %12 : tile<f32>
    }
    %2 = itof %assume_0 signed  : tile<i32> -> tile<f32>
    %reshape = reshape %2 : tile<f32> -> tile<1xf32>
    %3 = divf %reduce, %reshape  : tile<1xf32>
    %pview_20 = make_partition_view %tview_16 : partition_view<tile=(1), tensor_view<?xf32, strides=[?]>>
    %4 = store_view_tko weak %3, %pview_20[%blockId_x] token = %0 : tile<1xf32>, partition_view<tile=(1), tensor_view<?xf32, strides=[?]>>, tile<i32> -> token
    %cst_0_f32_21 = constant <f32: 0.000000e+00> : tile<1x4096xf32>
    %cst_0_i32_22 = constant <i32: 0> : tile<i32>
    %cst_1_i32_23 = constant <i32: 1> : tile<i32>
    %5 = iota : tile<4096xi32>
    %reshape_24 = reshape %assume_0 : tile<i32> -> tile<1xi32>
    %bcast = broadcast %reshape_24 : tile<1xi32> -> tile<4096xi32>
    %reshape_25 = reshape %3 : tile<1xf32> -> tile<1x1xf32>
    %bcast_26 = broadcast %reshape_25 : tile<1x1xf32> -> tile<1x4096xf32>
    %cst_0_f32_27 = constant <f32: 0.000000e+00> : tile<f32>
    %reshape_28 = reshape %cst_0_f32_27 : tile<f32> -> tile<1x1xf32>
    %bcast_29 = broadcast %reshape_28 : tile<1x1xf32> -> tile<1x4096xf32>
    %cst_2_f32 = constant <f32: 2.000000e+00> : tile<f32>
    %reshape_30 = reshape %cst_2_f32 : tile<f32> -> tile<1x1xf32>
    %bcast_31 = broadcast %reshape_30 : tile<1x1xf32> -> tile<1x4096xf32>
    %for_32 = for %loopIdx in (%cst_0_i32_22 to %1#1, step %cst_1_i32_23) : tile<i32> iter_values(%iterArg0 = %cst_0_f32_21) -> (tile<1x4096xf32>) {
      %pview_44 = make_partition_view %tview : partition_view<tile=(1x4096), padding_value = zero, tensor_view<?x?xf32, strides=[?,?]>>
      %tile, %result_token = load_view_tko weak %pview_44[%blockId_x, %loopIdx] token = %0 : partition_view<tile=(1x4096), padding_value = zero, tensor_view<?x?xf32, strides=[?,?]>>, tile<i32> -> tile<1x4096xf32>, token
      %12 = muli %loopIdx, %cst_4096_i32 : tile<i32>
      %reshape_45 = reshape %12 : tile<i32> -> tile<1xi32>
      %bcast_46 = broadcast %reshape_45 : tile<1xi32> -> tile<4096xi32>
      %13 = addi %bcast_46, %5 : tile<4096xi32>
      %14 = cmpi less_than %13, %bcast, signed : tile<4096xi32> -> tile<4096xi1>
      %15 = subf %tile, %bcast_26  : tile<1x4096xf32>
      %reshape_47 = reshape %14 : tile<4096xi1> -> tile<1x4096xi1>
      %16 = select %reshape_47, %15, %bcast_29 : tile<1x4096xi1>, tile<1x4096xf32>
      %17 = pow %16, %bcast_31 : tile<1x4096xf32>
      %18 = addf %iterArg0, %17  : tile<1x4096xf32>
      continue %18 : tile<1x4096xf32>
    }
    %reduce_33 = reduce %for_32 dim=1 identities=[0.000000e+00 : f32] : tile<1x4096xf32> -> tile<1xf32> 
    (%reduce_lhs: tile<f32>, %reduce_rhs: tile<f32>) {
      %12 = addf %reduce_lhs, %reduce_rhs  : tile<f32>
      yield %12 : tile<f32>
    }
    %6 = itof %assume_0 signed  : tile<i32> -> tile<f32>
    %reshape_34 = reshape %6 : tile<f32> -> tile<1xf32>
    %7 = divf %reduce_33, %reshape_34  : tile<1xf32>
    %reshape_35 = reshape %arg22 : tile<f32> -> tile<1xf32>
    %8 = addf %7, %reshape_35  : tile<1xf32>
    %9 = sqrt %8  : tile<1xf32>
    %cst_1_f32 = constant <f32: 1.000000e+00> : tile<f32>
    %reshape_36 = reshape %cst_1_f32 : tile<f32> -> tile<1xf32>
    %10 = divf %reshape_36, %9  : tile<1xf32>
    %pview_37 = make_partition_view %tview_19 : partition_view<tile=(1), tensor_view<?xf32, strides=[?]>>
    %11 = store_view_tko weak %10, %pview_37[%blockId_x] token = %0 : tile<1xf32>, partition_view<tile=(1), tensor_view<?xf32, strides=[?]>>, tile<i32> -> token
    %cst_0_i32_38 = constant <i32: 0> : tile<i32>
    %cst_1_i32_39 = constant <i32: 1> : tile<i32>
    %reshape_40 = reshape %3 : tile<1xf32> -> tile<1x1xf32>
    %bcast_41 = broadcast %reshape_40 : tile<1x1xf32> -> tile<1x4096xf32>
    %reshape_42 = reshape %10 : tile<1xf32> -> tile<1x1xf32>
    %bcast_43 = broadcast %reshape_42 : tile<1x1xf32> -> tile<1x4096xf32>
    for %loopIdx in (%cst_0_i32_38 to %1#1, step %cst_1_i32_39) : tile<i32> {
      %pview_44 = make_partition_view %tview : partition_view<tile=(1x4096), padding_value = zero, tensor_view<?x?xf32, strides=[?,?]>>
      %tile, %result_token = load_view_tko weak %pview_44[%blockId_x, %loopIdx] token = %0 : partition_view<tile=(1x4096), padding_value = zero, tensor_view<?x?xf32, strides=[?,?]>>, tile<i32> -> tile<1x4096xf32>, token
      %pview_45 = make_partition_view %tview_5 : partition_view<tile=(4096), padding_value = zero, tensor_view<?xf32, strides=[?]>>
      %tile_46, %result_token_47 = load_view_tko weak %pview_45[%loopIdx] token = %0 : partition_view<tile=(4096), padding_value = zero, tensor_view<?xf32, strides=[?]>>, tile<i32> -> tile<4096xf32>, token
      %pview_48 = make_partition_view %tview_8 : partition_view<tile=(4096), padding_value = zero, tensor_view<?xf32, strides=[?]>>
      %tile_49, %result_token_50 = load_view_tko weak %pview_48[%loopIdx] token = %0 : partition_view<tile=(4096), padding_value = zero, tensor_view<?xf32, strides=[?]>>, tile<i32> -> tile<4096xf32>, token
      %12 = subf %tile, %bcast_41  : tile<1x4096xf32>
      %13 = mulf %12, %bcast_43  : tile<1x4096xf32>
      %reshape_51 = reshape %tile_46 : tile<4096xf32> -> tile<1x4096xf32>
      %reshape_52 = reshape %tile_49 : tile<4096xf32> -> tile<1x4096xf32>
      %14 = fma %13, %reshape_51, %reshape_52  : tile<1x4096xf32>
      %pview_53 = make_partition_view %tview_13 : partition_view<tile=(1x4096), tensor_view<?x?xf32, strides=[?,?]>>
      %15 = store_view_tko weak %14, %pview_53[%blockId_x, %loopIdx] token = %0 : tile<1x4096xf32>, partition_view<tile=(1x4096), tensor_view<?x?xf32, strides=[?,?]>>, tile<i32> -> token
    }
    return
  }
}
