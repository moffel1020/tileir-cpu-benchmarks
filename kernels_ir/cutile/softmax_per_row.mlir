cuda_tile.module @kernels {
  entry @softmax_per_row(%arg0: tile<ptr<f32>>, %arg1: tile<i32>, %arg2: tile<i32>, %arg3: tile<i32>, %arg4: tile<i32>, %arg5: tile<ptr<f32>>, %arg6: tile<i32>, %arg7: tile<i32>, %arg8: tile<i32>, %arg9: tile<i32>) optimization_hints=<sm_89 = {}> {
    %0 = make_token : token
    %assume = assume bounded<0, ?>, %arg1 : tile<i32>
    %assume_0 = assume bounded<0, ?>, %arg2 : tile<i32>
    %tview = make_tensor_view %arg0, shape = [%assume, %assume_0], strides = [512, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[512,1]>
    %assume_1 = assume bounded<0, ?>, %arg6 : tile<i32>
    %assume_2 = assume bounded<0, ?>, %arg7 : tile<i32>
    %tview_3 = make_tensor_view %arg5, shape = [%assume_1, %assume_2], strides = [512, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[512,1]>
    %cst_512_i32 = constant <i32: 512> : tile<i32>
    %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
    %gridSize_x, %gridSize_y, %gridSize_z = get_num_tile_blocks : tile<i32>
    %cst_0_i32 = constant <i32: 0> : tile<i32>
    %cst_0_i32_4 = constant <i32: 0> : tile<i32>
    for %loopIdx in (%blockId_x to %cst_512_i32, step %gridSize_x) : tile<i32> {
      %pview = make_partition_view %tview : partition_view<tile=(1x1024), tensor_view<?x?xf32, strides=[512,1]>>
      %tile, %result_token = load_view_tko weak %pview[%loopIdx, %cst_0_i32] token = %0 : partition_view<tile=(1x1024), tensor_view<?x?xf32, strides=[512,1]>>, tile<i32> -> tile<1x1024xf32>, token
      %reduce = reduce %tile dim=1 identities=[0xFF800000 : f32] : tile<1x1024xf32> -> tile<1xf32> 
      (%reduce_lhs: tile<f32>, %reduce_rhs: tile<f32>) {
        %5 = maxf %reduce_lhs, %reduce_rhs : tile<f32>
        yield %5 : tile<f32>
      }
      %reshape = reshape %reduce : tile<1xf32> -> tile<1x1xf32>
      %bcast = broadcast %reshape : tile<1x1xf32> -> tile<1x1024xf32>
      %1 = subf %tile, %bcast  : tile<1x1024xf32>
      %2 = exp %1 : tile<1x1024xf32>
      %reduce_5 = reduce %2 dim=1 identities=[0.000000e+00 : f32] : tile<1x1024xf32> -> tile<1xf32> 
      (%reduce_lhs: tile<f32>, %reduce_rhs: tile<f32>) {
        %5 = addf %reduce_lhs, %reduce_rhs  : tile<f32>
        yield %5 : tile<f32>
      }
      %reshape_6 = reshape %reduce_5 : tile<1xf32> -> tile<1x1xf32>
      %bcast_7 = broadcast %reshape_6 : tile<1x1xf32> -> tile<1x1024xf32>
      %3 = divf %2, %bcast_7  : tile<1x1024xf32>
      %pview_8 = make_partition_view %tview_3 : partition_view<tile=(1x1024), tensor_view<?x?xf32, strides=[512,1]>>
      %4 = store_view_tko weak %3, %pview_8[%loopIdx, %cst_0_i32_4] token = %0 : tile<1x1024xf32>, partition_view<tile=(1x1024), tensor_view<?x?xf32, strides=[512,1]>>, tile<i32> -> token
    }
    return
  }
}
