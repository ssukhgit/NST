def style_transfer(contentImg, styleImg, alpha, wct):
    """
    Perform style transfer on a content image using a given style image.

    This function applies a sequence of encoder-decoder transformations at multiple levels
    (levels 5 to 1) using the provided WCT (Whitening and Coloring Transform) module. At each level,
    features are extracted, processed, and then decoded back, progressively transferring the style
    from the style image to the content image.

    Args:
        content_img (Tensor): The content image tensor.
        style_img (Tensor): The style image tensor.
        alpha (float): The interpolation factor controlling the intensity of style transfer.
        wct (Any): A module that implements the style transfer operations, with methods:
            - e1, e2, e3, e4, e5 for encoding,
            - d1, d2, d3, d4, d5 for decoding, and
            - transform for feature transformation.

    Returns:
        Tensor: The final stylized image tensor.
    """

    sF5 = wct.e5(styleImg)
    cF5 = wct.e5(contentImg)
    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    csF5 = wct.transform(cF5, sF5, alpha)
    Im5 = wct.d5(csF5)

    sF4 = wct.e4(styleImg)
    cF4 = wct.e4(Im5)
    sF4 = sF4.data.cpu().squeeze(0)
    cF4 = cF4.data.cpu().squeeze(0)
    csF4 = wct.transform(cF4, sF4, alpha)
    Im4 = wct.d4(csF4)

    sF3 = wct.e3(styleImg)
    cF3 = wct.e3(Im4)
    sF3 = sF3.data.cpu().squeeze(0)
    cF3 = cF3.data.cpu().squeeze(0)
    csF3 = wct.transform(cF3, sF3, alpha)
    Im3 = wct.d3(csF3)

    sF2 = wct.e2(styleImg)
    cF2 = wct.e2(Im3)
    sF2 = sF2.data.cpu().squeeze(0)
    cF2 = cF2.data.cpu().squeeze(0)
    csF2 = wct.transform(cF2, sF2, alpha)
    Im2 = wct.d2(csF2)

    sF1 = wct.e1(styleImg)
    cF1 = wct.e1(Im2)
    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    csF1 = wct.transform(cF1, sF1, alpha)
    Im1 = wct.d1(csF1)
    return Im1
