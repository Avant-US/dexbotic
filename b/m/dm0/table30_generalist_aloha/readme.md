词表和 table30_open_the_drawer 一样,但没有chat template(里面好像包含对tool的处理)
    add_tkn, mrge.txt, spcl_tkn_map, tkn_cfg.jsn, vcb
模型比 table30_open_the_drawer 多了 progress 的处理, transformers 版本高一点
    mdl.aftnsr.idx.jsn, cfg.jsn
词表和 table30_put_opener_in_drawer 一样,但没有chat template(里面好像包含对tool的处理)
模型比 table30_put_opener_in_drawer 多了 progress 的处理, transformers 版本高一点
词表和 table30_generalist_franka 一样
模型和 table30_generalist_franka 一样
词表和 base 一样,但没有chat template(里面好像包含对tool的处理), 而且model_max_length前者仅100后者4096. vcb后者压缩过,但猜测应一样.
模型和 base 从cfg.jsn看好像很大不同,但只是base没把默认值写上去,其实基本一致