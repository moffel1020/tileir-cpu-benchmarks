cuda_tile.module @kernels {
  entry @softmax_1d(%arg0: tile<ptr<f32>>, %arg1: tile<i32>, %arg2: tile<i32>, %arg3: tile<ptr<f32>>, %arg4: tile<i32>, %arg5: tile<i32>) optimization_hints=<sm_89 = {}> {
    %0 = make_token : token
    %assume = assume bounded<0, ?>, %arg1 : tile<i32>
    %assume_0 = assume bounded<0, ?>, %arg2 : tile<i32>
    %tview = make_tensor_view %arg0, shape = [%assume], strides = [%assume_0] : tile<i32> -> tensor_view<?xf32, strides=[?]>
    %assume_1 = assume bounded<0, ?>, %arg4 : tile<i32>
    %assume_2 = assume bounded<0, ?>, %arg5 : tile<i32>
    %tview_3 = make_tensor_view %arg3, shape = [%assume_1], strides = [%assume_2] : tile<i32> -> tensor_view<?xf32, strides=[?]>
    %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
    %pview = make_partition_view %tview : partition_view<tile=(128), tensor_view<?xf32, strides=[?]>>
    %tile, %result_token = load_view_tko weak %pview[%blockId_x] token = %0 : partition_view<tile=(128), tensor_view<?xf32, strides=[?]>>, tile<i32> -> tile<128xf32>, token
    %1 = exp %tile : tile<128xf32>
    %reduce = reduce %1 dim=0 identities=[0.000000e+00 : f32] : tile<128xf32> -> tile<f32> 
    (%reduce_lhs: tile<f32>, %reduce_rhs: tile<f32>) {
      %4 = addf %reduce_lhs, %reduce_rhs  : tile<f32>
      yield %4 : tile<f32>
    }
    %reshape = reshape %reduce : tile<f32> -> tile<1xf32>
    %bcast = broadcast %reshape : tile<1xf32> -> tile<128xf32>
    %2 = divf %1, %bcast  : tile<128xf32>
    %pview_4 = make_partition_view %tview_3 : partition_view<tile=(128), tensor_view<?xf32, strides=[?]>>
    %3 = store_view_tko weak %2, %pview_4[%blockId_x] token = %0 : tile<128xf32>, partition_view<tile=(128), tensor_view<?xf32, strides=[?]>>, tile<i32> -> token
    return
  }
}
