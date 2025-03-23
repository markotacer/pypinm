window.Global_Store = {
    galleryCart          : '.js-gal-cart',
    favGalleryCart       : '.js-fav-gal-cart',
    addPhoto             : '.js-add-photo',
    floatingCartOverlay  : '.js-proofing-fc-overlay',
    floatingCartBlock    : '.js-proofing-fc-block',
    floatingCartClose    : '.js-proofing-fc-close',
    cartItemRemove       : '.js-cart-item-remove',
    floatingCartBtn      : '.js-floating-pcart',
    insideCartQuantity   : '.js-inside-cart-quantity',
    insideStoreAddCart   : '.js-store-add-cart-quantity',
    checkoutForm         : '.js-pcheckout-form',
    cropImageBtn         : '.js-crop-image',
    cropImageCloseBtn    : '.js-image-crop-close-btn',
    cropImage            : '.js-image-crop-overlay',
    cropCartUpdate       : '.js-update-cart-img-coordinates',
  
    gridItem             : '.js-store-grid-item',
    addCartBtn           : '.js-addcart-btn',
    listingAddCartBtn    : '.js-listing-addcart-btn',
    productDetailAddCartBtn : '.js-storeproduct-add-cart-quantity',
  }
  
  Global_Store.Init = function() {
  
    Global_Store.cropModalInit();
  
    $(function(){
      // Global_Store.jCropint(this);
    });
  
    $.each($('.js-gal-cart-ref'), function( key, value ) {
      var s = '.js-'+$(value).data('imageid')+'-'+$(value).data('optionid');
      $(s).attr('data-quantity', $(value).data('quantity'));
    });
  
    $.each($('.js-fullgallery-cartitems'), function( key, value ) {
      // console.log("value--" , value);
      var qty = '.js-fullgallery-'+$(value).data('galleryid')+'-'+$(value).data('fullgalleryoptionid');
      // console.log("qty - ", qty);
      $(qty).attr('data-quantity', $(value).data('quantity'));
      $(qty).attr('data-fullgalleryoptionid', $(value).data('fullgalleryoptionid'));
      $(qty).attr('data-fullgalleryimageid', $(value).data('imageid'));
  
      // var galleryclass = $(value).attr('data-galleryclass');
      // console.log("galleryclass--" , galleryclass);
  
      // var galleryid = $(value).attr('data-galleryid');
      // console.log("galleryid--" , galleryid);
  
      // $.each($(".js-gal-cart-option").hasClass($(value).attr('data-galleryclass')), function( key, value ) {
      //   $(this).attr('data-quantity', '1');
      // });
  
      // $(galleryclass).attr('data-quantity', '1');
  
      // $('.js-fullgallery-'+galleryid).attr('data-quantity', $(value).data('quantity'));
    });
  
  
    var csrf_token, photoid, temp, temp1, image, title, quantity, price,
     counter, currency, itemId, gallerySlug, sampleText, type, itemadded,
      itemremove, crop_data, catalogid, printable, appendfavimg, notytitle, galleryid;
    if ($('#cart_csrf_token').length > 0) {
      var csrf_token = $('#cart_csrf_token').val();
  } else {
      var csrf_token = $('#csrf_token').val();
  }
  
    $('body').on('click', Global_Store.addPhoto, function (event) {
      var $that = $(this)
      $('.js-proofing-fc-loading').show();
      if( $(this).data('pagetype') == 'fav-page' )
        gallerySlug = $(this).data('linkslug');
      else
        gallerySlug = $('.js-price-list-body').attr('data-linkslug');
  
      galleryid = $('.js-price-list-img').attr('data-galleryid');
      image = $('.js-price-list-img img').attr('src');
      currency = $('.js-global-currency').val();
      itemadded = $('.js-price-list-body').data('itemadded');
      itemremove = $('.js-price-list-body').data('itemremove');
      crop_data = $(this).parent().find('span');
      label_id = $(this).data('labelid');
      downloadtype = $(this).data('downloadtype');
      // console.log("label_id" + galleryid + label_id);
      temp2 = '.js-add-fullgallery-'+galleryid+"-"+label_id;
      // console.log("temp2--" + temp2);
      appendfavimg = '<div class="js-favimg-noty favimg-noty"><img src="'+image+'"></div>';
      var data_x, data_y, data_h, data_w, data_oh, data_ow, priority, aspx, aspy;
      // data_x = parseInt(crop_data.attr('data-x')!="undefined"?crop_data.attr('data-x'):-1);
      // data_y = parseInt(crop_data.attr('data-y')!="undefined"?crop_data.attr('data-y'):-1);
      // data_h = parseInt(crop_data.attr('data-h')!="undefined"?crop_data.attr('data-h'):-1);
      // data_w = parseInt(crop_data.attr('data-w')!="undefined"?crop_data.attr('data-w'):-1);
      // data_oh = parseInt(crop_data.attr('data-oh')!="undefined"?crop_data.attr('data-oh'):-1);
      // data_ow = parseInt(crop_data.attr('data-ow')!="undefined"?crop_data.attr('data-ow'):-1);
      // aspx = crop_data.attr('data-aspx') != "undefined" ? crop_data.attr('data-aspx') : -1;
      // aspy = crop_data.attr('data-aspy') != "undefined" ? crop_data.attr('data-aspy') : -1;
      spanCount = parseInt($(this).siblings('span').text());
      price = $(this).data('price');
      title = $(this).data('title');
      // labelid = $(this).data('labelid');
      catalogid = $(this).data('labcatalogid');
      printable = $(this).attr('data-printable');
      type = $(this).data('type');
      // console.log('full id ', $(temp2).attr('data-fullgalleryoptionid'));
      if( downloadtype == 2 && !($(temp2).attr('data-fullgalleryoptionid') == '' || $(temp2).attr('data-fullgalleryoptionid') == 0)) {
        labelid = $(temp2).attr('data-fullgalleryoptionid');
        photoid = $(temp2).attr('data-fullgalleryimageid');
        // console.log("typeof enter-- " , type)
      } else {
        labelid = $(this).data('labelid');
        photoid = $('.js-price-list-img').attr('data-imageid');
        // console.log("typeof note enter-- " , type)
      }
  
      appendfavimg = '<div class="js-favimg-noty favimg-noty left"><img src="'+image+'"></div>';
      notytitle = '<div class="js-favimg-title noty-favimg-title">'+title+'</div>';
  
      if( type == 2 ) {
        if( $(this).hasClass('plus-icon') ) {
          if( spanCount+1 >= 2) {
            $('.js-proofing-fc-loading').hide();
            sweetAlert((typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_zero : "Oops..."), (typeof window.labels != 'undefined' ? window.labels.store_cart_digital_warning : "This digital item is already in your cart!"), "error");
            return false;
            return false;
          }
          quantity = 1;
          sampleText = itemadded;
          $(this).siblings('span').text(1);
          $(this).siblings('.js-minus-digital-photo').show();
          $(this).hide();
        } else {
  
          if( spanCount-1 < 0) {
            $('.js-proofing-fc-loading').hide();
            sweetAlert((typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_zero : "Oops..."), (typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_cannot_zero : "Quantity cannot be zero!"), "error");
            return false;
          } else {
            quantity = 0;
            sampleText = itemremove;
            $(this).siblings('span').text(0);
            $(this).siblings('.js-add-digital-photo').show();
            $(this).hide();
          }
        }
      } else {
        if( $(this).hasClass('plus-icon') ) {
          $(this).siblings('span').text(spanCount+1);
          quantity = spanCount+1;
          sampleText = itemadded;
        } else {
          if( (spanCount-1) < 0 ) {
            $('.js-proofing-fc-loading').hide();
            sweetAlert((typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_zero : "Oops..."), (typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_cannot_zero : "Quantity cannot be zero!"), "error");
            return false;
          } else {
            $(this).siblings('span').text(spanCount-1);
            quantity = spanCount-1;
          }
          sampleText = itemremove;
        }
      }
  
      temp1 = '.js-'+$('.js-price-list-img').attr('data-imageid')+'-'+labelid;
      // console.log("temp 1--" + '.js-'+$('.js-price-list-img').attr('data-imageid')+'-'+labelid);
      $(temp1).attr('data-quantity', quantity);
      $(temp2).attr('data-quantity', quantity);
      if (downloadtype == 2) {
        // console.log("prnce 1");
        if (quantity == 1) {
          $(temp2).attr('data-fullgalleryoptionid', labelid);
          $(temp2).attr('data-fullgalleryimageid', photoid);
          // console.log("prnce 2");
        } else{
          $(temp2).attr('data-fullgalleryoptionid', '0');
          $(temp2).attr('data-fullgalleryimageid', '0');
          // console.log("prnce 3");
        }
      }
  
      // $.each($('.js-fullgallery-'+galleryid), function( key, value ) {
      //   $(this).attr('data-quantity', quantity);
      // });
  
      //==================
      temp = $("#cart-item-template").html();
      temp = $(temp)[0];
      $(temp).attr('data-optionlabel', labelid);
      $(temp).attr('data-productid', photoid);
      $(temp).find('.cart-item-image').attr('src', image);
      $(temp).find('.cart-item-title').html(title);
      $(temp).find('.cart-item-label').hide();
      $(temp).find('.cart-item-unitprice').html(currency+price);
      $(temp).find('.cart-item-quantity').val(quantity);
      $(temp).find('.cart-item-quantity').attr('data-prevvalue', quantity);
      $(temp).find('.cart-item-quantity').attr('data-title', title);
      $(temp).find('.cart-item-quantity').attr('data-maxlimit', $('.js-quantity').attr('max'));
      $(temp).find('.cart-item-quantity').attr('data-type', type);
      $(temp).find('.cart-item-quantity').attr('data-product_type', 'proofing');
      if( type == 2 ){
        $(temp).find('.product_quantity_container').hide();      
        $(temp).find('.js-cart-item-remove').addClass('margin-l0');
        $(temp).find('.js-cart-item-remove').removeClass('menu-icon');
        $(temp).find('.js-cart-item-remove').text('Remove');
      }
  
      $(temp).find('.js-store-add-cart-quantity').attr({
        'data-prevvalue': quantity,
        'data-title': title,
        'data-maxlimit': $('.js-quantity').attr('max'),
        'data-type': type,
        'data-product_type': 'proofing'
      });
  
      $(temp).find('.js-cart-item-remove').attr('data-product_type', 'proofing');
      $(temp).find(Global_Store.cropImageBtn).attr('data-src',image);
      // $(temp).find('.cart-item-price').html(currency+''+price*quantity);
      $(temp).find('.cart-item-price').html(currency+''+parseFloat(price*quantity).toFixed(2)+"");
      // $(temp).find(Global_Store.cropImageBtn).attr('data-x',data_x);
      // $(temp).find(Global_Store.cropImageBtn).attr('data-y',data_y);
      // $(temp).find(Global_Store.cropImageBtn).attr('data-h',data_h);
      // $(temp).find(Global_Store.cropImageBtn).attr('data-w',data_w);
      // $(temp).find(Global_Store.cropImageBtn).attr('data-oh',data_oh);
      // $(temp).find(Global_Store.cropImageBtn).attr('data-ow',data_ow);
      // $(temp).find(Global_Store.cropImageBtn).attr('data-aspx',aspx);
      // $(temp).find(Global_Store.cropImageBtn).attr('data-aspy',aspy);
      // if (printable != 0) {
      //   $(temp).find('.jcrop-img-view').attr('data-x',data_x);
      //   $(temp).find('.jcrop-img-view').attr('data-y',data_y);
      //   $(temp).find('.jcrop-img-view').attr('data-h',data_h);
      //   $(temp).find('.jcrop-img-view').attr('data-w',data_w);
      //   $(temp).find('.jcrop-img-view').attr('data-oh',data_oh);
      //   $(temp).find('.jcrop-img-view').attr('data-ow',data_ow);
      //   $(temp).find('.jcrop-img-view').attr('data-aspx',aspx);
      //   $(temp).find('.jcrop-img-view').attr('data-aspy',aspy);
      //   $(function(){
      //     $(temp).find('.jcrop-img-view img').Jcrop({
      //       bgColor:     'black',
      //       bgOpacity:   .4,
      //       setSelect:   [ data_x, data_y, data_h, data_w ],
      //       aspectRatio: aspx / aspy,
      //       allowResize : false,
      //       allowSelect : false,
      //       boxWidth: 70, boxHeight: 105,
      //       trueSize: [data_w,data_h]
      //     });
      //   });
      // }
      // if( catalogid < 0 ) {
      //   $(temp).find(Global_Store.cropImageBtn).remove();
      // }
  
      if (printable == 0) {
        $(temp).find(Global_Store.cropImageBtn).remove();
        $(temp).find('.crop-image-view').removeClass('jcrop-img-view');
      }
  
      // console.log("printable " + printable);
      //==================
  
      (function(templ) {
        $.ajax({
          type: 'POST',
          url: $('.js-price-list-body').data('requestslug')+'?_token='+ csrf_token,
          data: {
            'id': photoid,
            'option': labelid,
            'quantity': quantity,
            'gallery-slug' : gallerySlug,
            'lab_catalog_id' : catalogid,
            // 'data_x': data_x,
            // 'data_y': data_y,
            // 'data_h': data_h,
            // 'data_w': data_w,
            // 'data_oh': data_oh,
            // 'data_ow': data_ow,
            // 'print_width': aspx,
            // 'print_height': aspy,
            'global': '1',
          }
        })
        .done(function(data) {
          // console.log(data);
          // console.log("aspy" + data.aspy + " aspx " + data.aspx  + " oh " + data.oh + " ow " + data.ow + " w " + data.w + " h " + data.h + " x " + data.x + " y " + data.y);
          $(temp).find(Global_Store.cropImageBtn).attr('data-x',data.x);
          $(temp).find(Global_Store.cropImageBtn).attr('data-y',data.y);
          $(temp).find(Global_Store.cropImageBtn).attr('data-h',data.h);
          $(temp).find(Global_Store.cropImageBtn).attr('data-w',data.w);
          $(temp).find(Global_Store.cropImageBtn).attr('data-oh',data.oh);
          $(temp).find(Global_Store.cropImageBtn).attr('data-ow',data.ow);
          $(temp).find(Global_Store.cropImageBtn).attr('data-aspx',data.aspx);
          $(temp).find(Global_Store.cropImageBtn).attr('data-aspy',data.aspy);
          if (printable != 0) {
            $(temp).find('.jcrop-img-view').attr('data-x',data.x);
            $(temp).find('.jcrop-img-view').attr('data-y',data.y);
            $(temp).find('.jcrop-img-view').attr('data-h',data.h);
            $(temp).find('.jcrop-img-view').attr('data-w',data.w);
            $(temp).find('.jcrop-img-view').attr('data-oh',data.oh);
            $(temp).find('.jcrop-img-view').attr('data-ow',data.ow);
            $(temp).find('.jcrop-img-view').attr('data-aspx',data.aspx);
            $(temp).find('.jcrop-img-view').attr('data-aspy',data.aspy);
            $(function(){
              $(temp).find('.jcrop-img-view img').Jcrop({
                bgColor:     'black',
                bgOpacity:   .4,
                setSelect:   [ data.x, data.y, data.h, data.w ],
                aspectRatio: data.aspx / data.aspy,
                allowResize : false,
                allowSelect : false,
                boxWidth: 70, boxHeight: 105,
                trueSize: [data.w,data.h]
              });
            });
          }
  
          $('.js-proofing-fc-loading').hide();
          if(photoid == data.productId)
            Global_Store.AddNewItem(templ);
  
          $('.js-ptotal-amount').html(data.totalPrice);
          $('.js-pcart-quantity').each(function() {
            if($(this).parent().find('.cart-label').length !== 0){
              $(this).text('('+data.quantity+')');
            } else{
              $(this).text(data.quantity);
            }
          })
  
          if( data.quantity == 0) {
            $('.js-floating-pcart-empty').show();
            $(Global_Store.floatingCartBtn).addClass('cart-not-active');
            $('.js-pcheckout-container').hide();
            $('.js-cart-qty-container').hide();
          } else {
            $('.js-floating-pcart-empty').hide();
            $('.js-pcheckout-container').show();
            $(Global_Store.floatingCartBtn).removeClass('cart-not-active');
            $('.js-proofing-header-cart').addClass('favourite-count-active');
            $('.js-cart-qty-container').show();
          }
  
          if( data.quantity == 0) {
            $('.js-floating-cart-empty').show();
            $('.js-checkout-container').hide();
            $('.js-floating-cart').addClass('cart-not-active');
          }
  
          if( data.cartImages[photoid] != undefined ) {
            $($('.js-slideshowgal-cart')[window.slideIndex]).addClass('favourite-info-active');
            $(".shopping-cart-show[data-imageid='"+photoid+"']").css("opacity", "1");
            $(".shopping-cart-show[data-imageid='"+photoid+"']").addClass("shopping-cart-active");
          } else {
            $(".shopping-cart-show[data-imageid='"+photoid+"']").css("opacity", "0");
            $(".shopping-cart-show[data-imageid='"+photoid+"']").removeClass("shopping-cart-active");
            $($('.js-slideshowgal-cart')[window.slideIndex]).removeClass('favourite-info-active');
          }
  
          if(parseInt(data.totalPrice) >= parseInt($('.js-price-list-body').attr('data-mincartprice')) ) {
            $(Global_Store.checkoutForm).show().removeClass('hide-all');
            $('.js-mincart-error').hide().addClass('hide-all');
          } else {
            $('.js-mincart-error').show().removeClass('hide-all');
            $(Global_Store.checkoutForm).hide().addClass('hide-all');
          }
          $.each($('.js-proofing-cart-item-listing .js-cart-item-list'), function( key, value ) {
            if( data.productId == $(value).attr('data-productid') && data.optionId == $(value).attr('data-optionlabel')) {
              $(value).find(Global_Store.cartItemRemove).attr('data-itemid', data.itemId);
              $(value).find(Global_Store.insideCartQuantity).attr('data-itemid', data.itemId);
              $(value).find('.js-store-add-cart-quantity').attr('data-itemid', data.itemId);
            }
          });
        });
      })(temp);
  
      Global_Store.AddNewItem = function(tmpl) {
        if($('.js-proofing-cart-item-listing .js-cart-item-list').length != 0) {
          $.each($('.js-proofing-cart-item-listing .js-cart-item-list'), function( key, value ) {
            if( $(value).attr('data-productid') == photoid && $(value).attr('data-optionlabel') == labelid) {
              $(value).remove();
              counter = 1;
            } else {
              counter = 1;
            }
          });
        } else {
          // $('.js-pcart-item-container').append(tmpl);
          $('.js-pcart-item-container').find('.js-proofing-cart-item-listing').append(tmpl);
        }
  
        if(counter == 1) {
          // $('.js-pcart-item-container').append(tmpl);
          $('.js-pcart-item-container').find('.js-proofing-cart-item-listing').append(tmpl);
        }
  
        $.each($('.js-proofing-cart-item-listing .js-cart-item-list'), function( key, value ) {
        if( $(value).find(Global_Store.insideCartQuantity).val() == 0 || $(value).find(Global_Store.insideCartQuantity).val() == '')
          $(value).remove();
        });
  
        var n = noty($.extend({
          text        : sampleText,
          type        : 'success',
          template    : '<div class="noty_message text-left left">'+appendfavimg+notytitle+'<span class="noty_text before-none"></span><div class="noty_close"></div></div>',
          timeout     :  4000,
        }, window.notyDefaults));
  
        Global_Store.cropModalInit();
  
        // $(function(){
        //   Global_Store.jCropint(this);
        // });
  
      }
    });
  
    $('body').on('change', Global_Store.insideCartQuantity, function (event) {
      var prev, product_type, quan = $(this).val();
      product_type = $(this).attr('data-product_type');
      // console.log("product_type " + product_type);
      if( quan <= 0) {
        sweetAlert((typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_zero : "Oops..."), (typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_cannot_zero : "Quantity cannot be zero!"), "error");
        $(this).val($(this).attr('data-prevvalue'));
        quan = $(this).attr('data-prevvalue');
      } else if( !Number.isInteger(parseFloat(quan)) || $(this).attr('data-type') == 2 ) {
        quan = 1;
        $(this).val(1);
      }
  
      itemId = $(this).data('itemid');
      var $that = $(this);
  
      temp1 = '.js-'+$that.parents('.js-cart-item-list').attr('data-productid')+'-'+$that.parents('.js-cart-item-list').attr('data-optionlabel');
      $(temp1).attr('data-quantity', quan);
  
      // console.log(quan);
      // console.log(itemId)
      if (product_type == 'proofing') {
        var inside_url = "/site/floating-quantity/"
      } else{
        var inside_url = "/floating-quantity/"
      }
  
      $.ajax({
          type: 'POST',
          url: inside_url + itemId +'?_token='+ csrf_token,
          data: {
            'quantity': quan,
            'global': '1',
            'client-slug': $('.js-price-list-body').attr('data-clientslug'),
          },
        })
      .done(function(data) {
        // console.log(data);
        $that.siblings('.js-cart-item-price').html($('.js-global-currency').val()+''+parseFloat(data.product_total).toFixed(2)+"");
        $('.js-ptotal-amount').html(data.totalPrice);
        // console.log('cart2')
        $('.js-pcart-quantity').each(function() {
          if($(this).parent().find('.cart-label').length !== 0){
            $(this).text('('+data.quantity+')');
          } else{
            $(this).text(data.quantity);
          }
        })
  
        if( data.quantity == 0) {
          $('.js-cart-qty-container').hide();
        } else{
          $('.js-cart-qty-container').show();
        }
  
        if( parseInt(data.totalPrice) >= parseInt($('.js-price-list-body').attr('data-mincartprice')) ) {
                $(Global_Store.checkoutForm).show().removeClass('hide-all');
  
          $('.js-mincart-error').hide().addClass('hide-all');
        } else {
          $('.js-mincart-error').show().removeClass('hide-all');
          $(Global_Store.checkoutForm).hide().addClass('hide-all');
        }
  
      });
    });
  
    $('body').on('click', Global_Store.insideStoreAddCart, function (event) {
      var prev, product_type, quan = parseInt($(this).siblings('input').val());
      product_type = $(this).attr('data-product_type');
      spanCount = parseInt($(this).siblings('input').val());
      maxLimitQty = $(this).attr('data-maxlimit');
      // console.log("product_type " + product_type);    
  
      if( $(this).hasClass('plus-icon') ) {      
        if ($(this).attr('data-type') == 2) {
          quan = 1;
          $(this).siblings('input').val(1);
        } else {
          quan = spanCount+1;
          if( parseInt(quan) > parseInt($(this).attr('data-maxlimit')) ) {
            sweetAlert("Oops...", "You can add only "+ parseInt($(this).attr('data-maxlimit'))+' '+$(this).attr('data-title')+ " to the Cart!", "error");
            return false;
          }
          $(this).siblings('input').val(spanCount+1);
        }
      } else {
        if( (spanCount-1) <= 0 ) {
          sweetAlert((typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_zero : "Oops..."), (typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_cannot_zero : "Quantity cannot be zero!"), "error");
          return false;
        } else if( !Number.isInteger(parseFloat(quan)) || $(this).attr('data-type') == 2) {
          quan = 1;
          $(this).siblings('input').val(1);
        } else{
          $(this).siblings('input').val(spanCount-1);
          quan = spanCount-1;
          // quan = 1;
          // $(this).siblings('input').val(1);
        }
      }
  
      // svg addClassname added
      var svgIcons = $(this).find('svg use').attr('xlink:href');
      $(this).css('pointer-events', 'none');
      $(this).find('svg use').attr('xlink:href', '');
      $(this).find('svg use').attr('xlink:href', '#cart-loading-pixpa-icon');
  
      if (quan == 1) {
        if( $(this).hasClass('fa-minus') ) {
          $(this).addClass('active')
        }
      } else{
        $(this).siblings('.fa-minus').removeClass('active')
      }
      
      itemId = $(this).data('itemid');
      var $that = $(this);    
      temp1 = '.js-'+$that.parents('.js-cart-item-list').attr('data-productid')+'-'+$that.parents('.js-cart-item-list').attr('data-optionlabel');
      $(temp1).attr('data-quantity', quan);
  
      if (product_type == 'proofing') {
        var inside_url = "/site/floating-quantity/"
      } else{
        var inside_url = "/floating-quantity/"
      }
  
      $.ajax({
          type: 'POST',
          url: inside_url + itemId +'?_token='+ csrf_token,
          data: {
            'quantity': quan,
            'global': '1',
            'client-slug': $('.js-price-list-body').attr('data-clientslug'),
          },
        })
      .done(function(data) {
        $that.parents('.product-details').find('.js-cart-item-price').html($('.js-global-currency').val()+''+parseFloat(data.product_total).toFixed(2)+"");
        $that.parents('tr').find('.js-cart-amount-price').html($('.js-global-currency').val()+''+parseFloat(data.product_total).toFixed(2)+"");
        $('.js-ptotal-amount').html(data.totalPrice);
        $('.js-pcart-quantity').each(function() {
          if($(this).parent().find('.cart-label').length !== 0){
            $(this).text('('+data.quantity+')');
          } else{
            $(this).text(data.quantity);
          }
        })
  
        if( data.quantity == 0) {
          $('.js-cart-qty-container').hide();
        } else{
          $('.js-cart-qty-container').show();
        }
  
        if( parseInt(data.totalPrice) >= parseInt($('.js-price-list-body').attr('data-mincartprice')) ) {
          $(Global_Store.checkoutForm).show().removeClass('hide-all');
          $('.js-mincart-error').hide().addClass('hide-all');
        } else {
          $('.js-mincart-error').show().removeClass('hide-all');
          $(Global_Store.checkoutForm).hide().addClass('hide-all');
        }
  
        // console.log('success');
        setTimeout(function() {
          $that.css('pointer-events', '');
          $that.find('svg use').attr('xlink:href', svgIcons);  
        }, 300);
        
  
      });
    });
  
    $('body').on('click', Global_Store.cartItemRemove, function (event) {
      itemId = $(this).data('itemid');
      var product_type = $(this).attr('data-product_type');
      var $that = $(this);
  
      // console.log("product_type " + product_type);
      var product_type = $(this).attr('data-product_type');
      if (product_type == 'proofing') {
        var inside_url = "/site/floating-remove-item/"
      } else{
        var inside_url = "/floating-remove-item/"
      }
  
      swal({
        title: $('.js-price-list-body').data('willremoved'),
        text: "",
        type: "warning",
        showCancelButton: true,
        confirmButtonColor: "#DD6B55",
        confirmButtonText: (typeof window.labels != 'undefined' ? window.labels.store_cart_confirm_remove : "Yes, remove it"),
        cancelButtonText: (typeof window.labels != 'undefined' ? window.labels.store_cart_cancel_button : "Cancel"),
        closeOnConfirm: true,
        closeOnCancel: false,
        allowOutsideClick: true,
        showLoaderOnConfirm: true,
      }, function (isConfirm) {
        if(isConfirm) {
          $('button.confirm').attr('disabled', true);
          $.ajax({
              type: 'POST',
              url: inside_url + itemId +'?_token='+ csrf_token,
              data: {
                'global': '1',
              },
            })
          .done(function(data) {
            // console.log(data);
            // swal((typeof window.labels != 'undefined' ? window.labels.store_cart_itemdeleted : "Deleted!"), $('.js-price-list-body').data('itemremove'), "success");
            $('button.confirm').attr('disabled', false);
            $that.parents('.js-cart-item-list').remove();
            $('.js-ptotal-amount').html(data.totalPrice);
            $('.js-pcart-quantity').each(function() {
              if($(this).parent().find('.cart-label').length !== 0){
                $(this).text('('+data.quantity+')');
              } else{
                $(this).text(data.quantity);
              }
            })
  
  
            // console.log("product_type " + product_type)
  
            if (product_type == 'proofing') {
  
              var photoid = $that.parents('.js-cart-item-list').attr('data-productid');
              if( data.cartImages[$that.parents('.js-cart-item-list').attr('data-productid')] != undefined )
                $($('.js-slideshowgal-cart')[window.slideIndex]).addClass('favourite-info-active');
              else
                $($('.js-slideshowgal-cart')[window.slideIndex]).removeClass('favourite-info-active');
  
              if( parseInt(data.totalPrice) >= parseInt($('.js-price-list-body').attr('data-mincartprice')) ) {
                $(Global_Store.checkoutForm).show().removeClass('hide-all');
  
                $('.js-mincart-error').hide().addClass('hide-all');
              } else {
                $('.js-mincart-error').show().removeClass('hide-all');
                $(Global_Store.checkoutForm).hide().addClass('hide-all');
              }
  
              temp1 = '.js-'+$that.parents('.js-cart-item-list').attr('data-productid')+'-'+$that.parents('.js-cart-item-list').attr('data-optionlabel');
              $(temp1).attr('data-quantity', 0);
  
  
            } else{
              if( data.quantity > 1)
                $('.js-cart-value-label').text($('.js-itemlabel-holder').data('itemslabel'));
              else
                $('.js-cart-value-label').text($('.js-itemlabel-holder').data('itemlabel'));
  
  
              if( parseInt(data.totalPrice) >= parseInt($('.js-price-list-body').attr('data-mincartprice')) ) {
                $(Global_Store.checkoutForm).show().removeClass('hide-all');
  
                $('.js-mincart-error').hide().addClass('hide-all');
              } else {
                $('.js-mincart-error').show().removeClass('hide-all');
                $(Global_Store.checkoutForm).hide().addClass('hide-all');
              }
  
            }
  
  
  
            if( data.quantity == 0) {
              $('.js-floating-pcart-empty').show();
              $('.js-pcheckout-container').hide();
              $(".shopping-cart-show[data-imageid='"+photoid+"']").css("opacity", "0");
              $(Global_Store.floatingCartBtn).addClass('cart-not-active');
  
              $('.js-cart-qty-container').hide();
            } else{
              $('.js-cart-qty-container').show();
            }
  
            var n = noty($.extend({
              text        : $('.js-price-list-body').data('itemremove'),
              type        : 'success',
              template    : '<div class="noty_message"><span class="noty_text before-none"></span><div class="noty_close"></div></div>',
              timeout     :  6000,
            }, window.notyDefaults));
  
          });
        } else {
          swal((typeof window.labels != 'undefined' ? window.labels.store_cart_cancelled : "Cancelled"), (typeof window.labels != 'undefined' ? window.labels.store_cart_still_in_cart : "Item is still in cart :)"), "error");
          $('button.confirm').attr('disabled', false);
        }
      });
  
    });
  
    $('body').on('click', Global_Store.galleryCart, function (event) {
      event.stopPropagation();
      $('body').addClass('no-scroll');
      if (document.documentElement.scrollHeight > document.documentElement.clientHeight){
        $('body').css({
          'Width': 'calc (100% - 15px);'
        });
      } else {
        $('body').css({
          'Width': '100%'
        });
      }
      // $('.js-plt').hide();
      // $('.js-plt_'+$(this).attr('data-galleryid')).show();
  
      if ($(this).attr('data-title') != '' && typeof $(this).attr('data-title') !== "undefined") {
        $('.js-floating-img-title').html(' - ' + $(this).attr('data-title'));
      } else{
        $('.js-floating-img-title').html(' ');
      }
  
      $('.js-price-list-body').show();
      $('.js-pcart-item-wrapper').hide();
      $('.js-price-list-img').attr('data-imageid', $(this).attr('data-imageid'));
      $('.js-price-list-img').attr('data-galleryid', $(this).attr('data-galleryid'));
      $('.js-price-list-img img').attr('src', $(this).attr('data-smallscreen'));
      var original_height = $("[data-gridimageid='"+$(this).data('imageid')+"']").attr("data-height");
      var original_width = $("[data-gridimageid='"+$(this).data('imageid')+"']").attr("data-width");
      var h = original_height;
      var w = original_width;
      var as;
  
      if (typeof original_height != 'undefined' && typeof original_width != 'undefined') {
        w  = original_width;
        h = original_height;
  
        newUnit   = 500;
        if ($(this).attr('data-jcropnewimg') == "1") {
          newUnit   = 500;
        } else{
          newUnit   = 400;
        }
  
        if ( w <= newUnit && h <= newUnit) {
        } else {
  
          if (newUnit == 500) {
            if (w > newUnit) {
              w     =   newUnit;
              h    =  Math.round((newUnit*original_height)/original_width);
  
            }
  
            if (h > newUnit) {
  
              h    =   newUnit;
              w     =  Math.round((newUnit*original_width)/original_height);
            }
  
          } else {
  
            if (w > newUnit) {
              w     =   newUnit;
              h    =  Math.round((newUnit*original_height)/original_width);
  
            } else if (h > newUnit) {
  
              h    =   newUnit;
              w     =  Math.round((newUnit*original_width)/original_height);
            }
          }
  
  
        }
      }
  
      // console.log('height: '+h+'width: '+w + ' newUnit:' + newUnit);
  
      $.each($('.js-photono-'+$(this).attr('data-imageid')), function( key, value ) {
        $that = $(value);
        $.each($('.js-table-tr'), function( key, value ) {
          if( $that.attr('data-optionid') == $(value).attr('data-labelid') ) {
            $(value).find('span').text($that.attr('data-quantity'));
  
            if($that.attr('data-quantity') == 1){
              $(value).find('.js-minus-digital-photo').show();
              $(value).find('.js-add-digital-photo').hide();
            } else{
              $(value).find('.js-minus-digital-photo').hide();
              $(value).find('.js-add-digital-photo').show();
            }
  
  
          //=================== function to check if image has jcrop compatible aspect ratio
          //width, height, aspect ratio
          if( $(this).attr('data-labcatlogid') >= 0 && $(value).attr('data-download') == 0 ) {
            var div = $(value).find('.label-v');
            as = "";
            if( parseFloat($(div).attr("data-height")) > 0 && parseFloat($(div).attr("data-width")) > 0 ) {
              as = $(div).attr("data-width") +" x "+$(div).attr("data-height");
            } else {
              as = $(value).find('.label-v').text();
              if( as.indexOf('8 Up Wallet') !== -1 )
                as = "2.5 x 3.5";
            }
            // console.log("Aspect ratio: "+as);
            var res = checkAspectRatio(original_height, original_width, w, h, as);
            if(res.status) {
              $(value).show();
            } else {
              if ($that.attr('data-photocheck') == 'js-check-printable-size') {
                $(value).hide();
              }
            }
  
            // $(value).find('span').attr('data-x', res.x);
            // $(value).find('span').attr('data-y', res.y);
            // $(value).find('span').attr('data-h', res.h);
            // $(value).find('span').attr('data-w', res.w);
            // $(value).find('span').attr('data-oh', res.oh);
            // $(value).find('span').attr('data-ow', res.ow);
            // $(value).find('span').attr('data-aspx', res.aspx);
            // $(value).find('span').attr('data-aspy', res.aspy);
          }
          //=================== End
          }
        });
  
  
        // code to hide/show table headings
        $(".js-table-container").each(function(i, v){
          if($(v).find('.js-table-tr').is(':visible')) {
            $(v).find('.tr-heading').show();
  
            // prince code start
            var proid = $(v).find('.tr-heading').find('.label-v').attr('id');
            $(".proofing-category").find('.data-target').each(function() {
              if ($(this).attr('data-id') == proid) {
                $(this).show();
              }
            });
  
  
          } else {
            $(v).find('.tr-heading').hide();
  
            // price code start
            var proid = $(v).find('.tr-heading').find('.label-v').attr('id');
            $(".proofing-category").find('.data-target').each(function() {
              if ($(this).attr('data-id') == proid) {
                $(this).hide();
              }
            });
          }
  
  
        });
  
        if($('.js-table-tr').is(':visible')) {
          $(".js-options-not-found").hide();
        } else {
          $(".js-options-not-found").show();
        }
      });
  
      $('.js-price-list-img img').unbind('load');
      $('.js-price-list-img img').on('load', function() {
        $(Global_Store.floatingCartOverlay+' ,'+Global_Store.floatingCartBlock).addClass('active');
      });
    });
  
    //WHCC prints standards: 300dpi
    function dpiCheck(oh, ow, asx, asy) {
      // console.log('original_h: '+oh+" original_w: "+ow+" asp_x: "+asx+" asp_y: "+asy);
      if ((asx*300) <= ow && (asy*300) <= oh)
        return true;
      else
        return false;
    }
  
    function checkAspectRatio(oh, ow, w, h, as) {
      //console.log("In checkAspectRatio: inputs: "+w+", "+h+", "+as);
      var asp = as.toLowerCase().split("x");
      var x, y, priority;
      asp[0] = parseInt(asp[0]);
      asp[1] = parseInt(asp[1]);
      // console.log("height: "+oh+" width: "+ow);
  
      if( parseInt(oh) >= parseInt(ow) ) {
        y = ( asp[0] > asp[1] ? asp[0] : asp[1] );
        x = ( asp[0] < asp[1] ? asp[0] : asp[1] );
        priority = "fullheight";
      } else {
        x = ( asp[0] > asp[1] ? asp[0] : asp[1] );
        y = ( asp[0] < asp[1] ? asp[0] : asp[1] );
        priority = "fullwidth";
      }
  
      // console.log("priority: "+priority);
      // console.log("priority x: "+x);
      // console.log("priority y: "+y);
  
      window.CurrentImage = { large_height : oh, large_width : ow, height: h, width: w };
  
      w = parseInt(w);
      h = parseInt(h);
  
      var res = { priority: priority , aspx : x, aspy : y };
      res.status = false;
  
      if($(".js-printable-check-val").attr('data-photocheck') == 1){
        // console.log("printable 2--");
        if(!dpiCheck(oh, ow, x, y)) {
          res.status = false;
          res.x = -1;
          res.y = -1;
          res.h = -1;
          res.w = -1;
          res.ow = -1;
          res.oh = -1;
          return res;
        }
      }
      // if(!dpiCheck(oh, ow, x, y)) {
      //   res.status = false;
      //   res.x = -1;
      //   res.y = -1;
      //   res.h = -1;
      //   res.w = -1;
      //   res.ow = -1;
      //   res.oh = -1;
      //   return res;
      // }
  
      if( priority == "fullheight" ) {
  
        // console.log("In fullHeight");
  
        //=================== 100% height
        res.h = h;
        res.w = Math.round((h*x)/y);
        if (res.w <= w) {
          res.type = "height 100%";
          res.x = Math.round( ( w - res.w ) / 2 );
          res.y = 0;
          res.ow = res.w;
          res.oh = res.h;
          res.w += res.x;
          res.status = true;
          return res;
        }
        //==================== 100% width
        res.w = w;
        res.h = Math.round((w*y)/x);
        if (res.h <= h) {
          res.type = "width 100%";
          res.x = 0;
          res.y = Math.round( ( h - res.h ) / 2 );
          res.oh = res.h;
          res.ow = res.w;
          res.h += res.y;
          res.status = true;
          return res;
        }
      } else {
        // console.log("In fullWidth");
        //==================== 100% width
        res.w = w;
        res.h = Math.round((w*y)/x);
        if (res.h <= h) {
          res.type = "width 100%";
          res.x = 0;
          res.y = Math.round( ( h - res.h ) / 2 );
          res.oh = res.h;
          res.ow = res.w;
          res.h += res.y;
          res.status = true;
          return res;
        }
        //=================== 100% height
        res.h = h;
        res.w = Math.round((h*x)/y);
        if (res.w <= w) {
          res.type = "height 100%";
          res.x = Math.round( ( w - res.w ) / 2 );
          res.y = 0;
          res.ow = res.w;
          res.oh = res.h;
          res.w += res.x;
          res.status = true;
          return res;
        }
      }
  
      if ($(".js-printable-check-val").attr('data-photocheck') == 0) {
        // console.log("printable 1--");
        if(!dpiCheck(oh, ow, x, y)) {
          res.status = false;
          res.x = x;
          res.y = y;
          res.h = h;
          res.w = w;
          res.ow = ow;
          res.oh = oh;
          return res;
        }
      }
  
      return res;
      //====================
    }
  
    Global_Store.rotateCrop = function() {
      var item = _JCROP.active;
      var x, y, w, h, priority;
      var res = {};
  
      // x = ( ( $(item).attr("data-aspy") != "undefined" && parseInt($(item).attr("data-aspy")) > 0 ) ? parseInt($(item).attr("data-aspy")) :  );
      // y = ( ( $(item).attr("data-aspx") != "undefined" && parseInt($(item).attr("data-aspx")) > 0 ) ? parseInt($(item).attr("data-aspx")) :  );
      x = parseInt($(item).attr("data-aspy"));
      y = parseInt($(item).attr("data-aspx"));
  
      w = _JCROP.img.x;
      h = _JCROP.img.y;
  
      if(w > h) {
        priority = "fullwidth";
      } else {
        priority = "fullheight";
      }
  
      if( priority == "fullheight" ) {
        //=================== 100% height
        res.h = h;
        res.w = Math.round((h*x)/y);
        if (res.w <= w) {
          res.type = "height 100%";
          res.x = Math.round( ( w - res.w ) / 2 );
          res.y = 0;
          res.ow = res.w;
          res.oh = res.h;
          res.w += res.x;
          res.status = true;
        }
        //==================== 100% width
        res.w = w;
        res.h = Math.round((w*y)/x);
        if (res.h <= h) {
          res.type = "width 100%";
          res.x = 0;
          res.y = Math.round( ( h - res.h ) / 2 );
          res.oh = res.h;
          res.ow = res.w;
          res.h += res.y;
          res.status = true;
        }
      } else {
        //==================== 100% width
        res.w = w;
        res.h = Math.round((w*y)/x);
        if (res.h <= h) {
          res.type = "width 100%";
          res.x = 0;
          res.y = Math.round( ( h - res.h ) / 2 );
          res.oh = res.h;
          res.ow = res.w;
          res.h += res.y;
          res.status = true;
        }
        //=================== 100% height
        res.h = h;
        res.w = Math.round((h*x)/y);
        if (res.w <= w) {
          res.type = "height 100%";
          res.x = Math.round( ( w - res.w ) / 2 );
          res.y = 0;
          res.ow = res.w;
          res.oh = res.h;
          res.w += res.x;
          res.status = true;
        }
      }
  
      $(item).attr("data-x", res.x);
      $(item).attr("data-y", res.y);
      $(item).attr("data-h", res.h);
      $(item).attr("data-w", res.w);
      $(item).attr("data-oh", res.oh);
      $(item).attr("data-ow", res.ow);
      $(item).attr("data-aspx", x);
      $(item).attr("data-aspy", y);
      _JCROP.asp = { x : parseFloat($(item).attr('data-aspx')), y : parseFloat($(item).attr('data-aspy')) };
  
      _JCROP.jcrop.release();
      _JCROP.jcrop.setSelect([res.x, res.y, res.w, res.h]);
    }
  
    $('body').on('click', Global_Store.favGalleryCart, function (event) {
      event.stopPropagation();
      $('body').addClass('no-scroll');
      if (document.documentElement.scrollHeight > document.documentElement.clientHeight){
        $('body').css({
          'Width': 'calc (100% - 15px);'
        });
      } else {
        $('body').css({
          'Width': '100%'
        });
      }
      $('.js-plt').hide();
      $('.js-plt_'+$(this).attr('data-galleryid')).show();
      $('.js-price-list-body').show();
      $('.js-pcart-item-wrapper').hide();
      $('.js-price-list-img').attr('data-imageid', $(this).attr('data-imageid'));
      $('.js-price-list-img').attr('data-galleryid', $(this).attr('data-galleryid'));
      $('.js-price-list-img img').attr('src', $(this).attr('data-smallscreen'));
      var current_image_id = $(this).attr('data-imageid');
      var original_height = $("[data-gridimageid='"+$(this).data('imageid')+"']").attr("data-height");
      var original_width = $("[data-gridimageid='"+$(this).data('imageid')+"']").attr("data-width");
      var h, w, as;
  
      if (typeof original_height != 'undefined' && typeof original_width != 'undefined') {
        w  = original_width;
        h = original_height;
        newUnit   = 500;
        if ( w <= newUnit && h <= newUnit) {
        } else {
          if (w > newUnit) {
            w     =   newUnit;
            h    =  Math.round((newUnit*original_height)/original_width);
          }
          // if (h > newUnit) {
          //   h    =   newUnit;
          //   w     =  Math.round((newUnit*original_width)/original_height);
          // }
        }
      }
  
      // console.log('height: '+h+'width: '+w);
      // console.log('.js-photono-'+$(this).attr('data-imageid'));
  
      var galleryid = $(".price-list-table:visible").attr('data-galleryid');
      var iterator = $('.js-photono-'+$(this).attr('data-imageid'));
      if(typeof galleryid != "undefined" && galleryid != "" && $('.js-photono-'+$(this).attr('data-imageid')+'[data-galleryid='+galleryid+']').length > 0) {// in case of search blade
        // console.log("search blade");
        iterator = $('.js-photono-'+$(this).attr('data-imageid')+'[data-galleryid='+galleryid+']');
      }
      $.each(iterator, function( key, value ) {
        $that = $(value);
        $.each($('.js-table-tr'), function( key, value ) {
          if( $that.attr('data-optionid') == $(value).attr('data-labelid') ) {
            $(value).find('span').text($that.attr('data-quantity'));
  
            if($that.attr('data-quantity') == 1){
              $(value).find('.js-minus-digital-photo').show();
              $(value).find('.js-add-digital-photo').hide();
            } else{
              $(value).find('.js-minus-digital-photo').hide();
              $(value).find('.js-add-digital-photo').show();
            }
  
  
          //=================== function to check if image has jcrop compatible aspect ratio
          //width, height, aspect ratio
          // if( $(this).attr('data-labcatlogid') != 0 ) {
          if( $(this).attr('data-labcatlogid') >= 0 && $(value).attr('data-download') == 0 ) {
            var div = $(value).find('.label-v');
            as = "";
            if( parseFloat($(div).attr("data-height")) > 0 && parseFloat($(div).attr("data-width")) > 0 ) {
              as = $(div).attr("data-width") +" x "+$(div).attr("data-height");
            } else {
              as = $(value).find('.label-v').text();
              if( as.indexOf('8 Up Wallet') !== -1 )
                as = "2.5 x 3.5";
            }
            // console.log("Aspect ratio: "+as);
  
            var res = checkAspectRatio(original_height, original_width, w, h, as);
            if(res.status) {
              $(value).show();
            } else {
              // $(value).hide();
              if ($that.attr('data-photocheck') == 'js-check-printable-size') {
                $(value).hide();
              }
  
            }
            $(value).find('span').attr('data-x', res.x);
            $(value).find('span').attr('data-y', res.y);
            $(value).find('span').attr('data-h', res.h);
            $(value).find('span').attr('data-w', res.w);
            $(value).find('span').attr('data-oh', res.oh);
            $(value).find('span').attr('data-ow', res.ow);
            $(value).find('span').attr('data-aspx', res.aspx);
            $(value).find('span').attr('data-aspy', res.aspy);
          }
          //=================== End
          }
        });
      });
      // code to hide/show table headings
      $(".js-table-container").each(function(i, v) {
        if($(v).find('.js-table-tr').is(':visible')) {
          $(v).find('.tr-heading').show();
          // prince code start
          var proid = $(v).find('.tr-heading').find('.label-v').attr('id');
          console.log('proid1 ' + proid)
          $(".proofing-category").find('.data-target').each(function() {
            if ($(this).attr('data-id') == proid) {
              $(this).show();
            }
          });
        } else {
          $(v).find('.tr-heading').hide();
          // price code start
          var proid = $(v).find('.tr-heading').find('.label-v').attr('id');
          console.log('proid2 ' + proid)
          $(".proofing-category").find('.data-target').each(function() {
            if ($(this).attr('data-id') == proid) {
              $(this).hide();
            }
          }); 
        }
      });
  
      if($('.js-table-tr').is(':visible')) {
        $(".js-options-not-found").hide();
      } else {
        $(".js-options-not-found").show();
      }
  
      $('.js-price-list-img img').unbind('load');
      $('.js-price-list-img img').on('load', function() {
        $(Global_Store.floatingCartOverlay+' ,'+Global_Store.floatingCartBlock).addClass('active');
      });
    });
  
    $(Global_Store.floatingCartClose+' ,'+Global_Store.floatingCartOverlay).on('click', function(event) {
      $(Global_Store.floatingCartOverlay+' ,'+Global_Store.floatingCartBlock).removeClass('active');
      $('body').removeClass('no-scroll');
      $('body').css({'width': ''});
      $('.js-prod-desc-single-modal, .js-prod-info').removeClass('active');
      $('.js-prod-info').parent().parent('.js-table-tr').removeClass('active');
  
      $('.js-image-crop-close-btn').click();
  
    });
  
    $(Global_Store.floatingCartBtn+', .js-pcart-nextstep').unbind('click');
    $('body').on('click', Global_Store.floatingCartBtn+', .js-pcart-nextstep', function(event) {
      $(Global_Store.floatingCartOverlay+' ,'+Global_Store.floatingCartBlock).addClass('active');
      $('.js-price-list-body').hide();
      $('.js-pcart-item-wrapper').show();
      $('body').addClass('no-scroll');
      if (document.documentElement.scrollHeight > document.documentElement.clientHeight){
        $('body').css({
          'Width': 'calc (100% - 15px);'
        });
      } else {
        $('body').css({
          'Width': '100%'
        });
      }
    });
  
  
    // store js start
    $(Global_Store.gridItem).on('click', function(event) {
      window.location = $(this).data('url');
    });
  
  
    // $(".js-option-label").on('change', function(event) {
    //   if( $('.js-option-label option:selected').attr('data-quancheck') == 0 ) {
    //     $('.quantity, .js-addcart-btn').removeClass('hide-all');
    //     $('.quantity').children('.js-quantity').val(1);
    //     $('.quantity').children('.js-quantity').attr('max', $('.js-option-label option:selected').attr('data-quantity'));
    //     if($('.js-option-label option:selected').attr('data-before-sale-price') > 0) {
    //       var prevPrice = "&nbsp;&nbsp;<strike>"+$('.js-global-currency').val()+$('.js-option-label option:selected').attr('data-before-sale-price')+"</strike>";
    //       $('.js-base-price span').html($('.js-global-currency').val()+$('.js-option-label option:selected').attr('data-price') + prevPrice );
    //     } else {
    //       $('.js-base-price span').html($('.js-global-currency').val()+$('.js-option-label option:selected').attr('data-price'));
    //     }
    //   } else {
    //     $('.quantity, .js-addcart-btn').addClass('hide-all');
    //     $('.js-base-price span').html('Sold Out <strike>'+$('.js-global-currency').val()+$('.js-option-label option:selected').attr('data-price')+'</strike>');
    //   }
    // });
  
    $(".js-listing-option-label").on('change', function(event) {
      // console.log($('option:selected',this).attr('data-quancheck'));
      // console.log($('option:selected',this).attr('data-quantity'));
      if ($('option:selected',this).attr('data-active') == 0) {
        $(this).parents('.js-product-listing-details').find('.js-listing-addcart input').addClass('hide-all');
        $(this).parents('.grid-item').find('.js-base-price span').html('Unavailable');
      } else if( $('option:selected',this).attr('data-quancheck') == 0 ) {
        let variantTitles = $('option:selected',this).attr('data-variant_titles');
        $(this).parents('.js-product-listing-details').find('.js-listing-addcart input').removeClass('hide-all');
        $('.quantity').children('.js-quantity').val(1);
        $('.quantity').children('.js-quantity').attr('max', $('.js-option-label option:selected').attr('data-quantity'));
        if($('option:selected',this).attr('data-before-sale-price') > 0) {
          var prevPrice = "&nbsp;&nbsp;<strike>"+$('.js-global-currency').val()+$('option:selected',this).attr('data-before-sale-price')+"</strike>";
          $(this).parents('.grid-item').find('.js-base-price span').html($('.js-global-currency').val()+$('option:selected',this).attr('data-price') + prevPrice );
        } else {
          $(this).parents('.grid-item').find('.js-base-price span').html($('.js-global-currency').val()+$('option:selected',this).attr('data-price'));
        }
        $(this).parents('.js-product-listing-details').find('.js-listing-addcart input').attr('data-variant_titles', variantTitles);
      } else {
        $(this).parents('.js-product-listing-details').find('.js-listing-addcart input').addClass('hide-all');
        $(this).parents('.grid-item').find('.js-base-price span').html('Sold Out <strike>'+$('.js-global-currency').val()+$('option:selected',this).attr('data-price')+'</strike>');
      }
    });
  
    $(Global_Store.addCartBtn).unbind('click');
    $(Global_Store.addCartBtn).on('click', function(event) {
      $that = $(this);
      // console.log('$that', $that.parents('.product-container').find('.js-quantity'))
      productId = $(this).data('pid');
      image = window.productCartImage ? window.productCartImage : $(this).data('img');
      title = $(this).data('title');
      let productVariationData;
      if (typeof AllProductVariation === 'undefined') {
        productVariationData = typeof ProductVariation !== 'undefined' ? ProductVariation : undefined 
      } else {
        AllProductVariation.forEach(function(p) {
          if (p.productid == productId) {
            productVariationData = p
          }
        })
      }
      // console.log('productVariationData', productVariationData)
      labelId = typeof productVariationData !== 'undefined' ? productVariationData.selectedVariant.id : 0; // $('.js-option-label').val();
      quantity_threshold = $(this).data('quantity_threshold');
      // console.log(productId);
      $quantityDiv = $that.parents('.product-container').find('.js-quantity')
      $basepriceDiv = $that.parents('.product-container').find('.js-base-price')
      quantity = $quantityDiv.val();
      maxquantity = $quantityDiv.attr('max');
      itemadded = $('.js-store-cart-item-listing').data('itemadded');
  
      appendfavimg = '<div class="js-favimg-noty favimg-noty left"><img src="'+image+'"></div>';
      notytitle = '<div class="js-favimg-title noty-favimg-title">'+title+'</div>';
  
      // console.log( Number.isInteger(parseFloat(quantity)) );
      if( quantity <= 0) {
        sweetAlert((typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_zero : "Oops..."), (typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_cannot_zero : "Quantity cannot be zero!"), "error");
        return false;
      } else if( parseInt(quantity) > parseInt(maxquantity) ) {
        sweetAlert("Oops...", "You can add only "+ parseInt(maxquantity)+' '+title+ " to the Cart!", "error");
        return false;
      } else if( !Number.isInteger(parseFloat(quantity)) ) {
        // console.log(quantity);
        quantity = 1;
        $quantityDiv.val(1);
      }
  
  
      $('.js-floating-pcart-empty').hide();
      $('.js-checkout-container').show();
      $('.js-floating-cart').removeClass('cart-not-active');
  
      if(labelId == undefined)
        labelId = 0;
  
      // console.log(labelId);
      // console.log(productId);
  
      if(typeof productVariationData !== 'undefined' && labelId)
        price = productVariationData.selectedVariant.price;
      else
        price = $basepriceDiv.data('baseprice');
  
      currency = $('.js-global-currency').val();
      labelValue = typeof productVariationData !== 'undefined' ? productVariationData.selectedVariant.label : '';
      counter = 0;
  
  
      $(this).val('ADDING...');
  
      $.ajax({
          type: 'POST',
          url: '/cart?_token=' + csrf_token,
          data: {
            'option': labelId,
            'quantity': quantity,
            'productid':productId,
            'global' : '1',
          }
        })
      .done(function(data) {
        if( parseInt(data.totalPrice) >= parseInt($('.js-price-list-body').attr('data-mincartprice')) ) {
          $(Global_Store.checkoutForm).show().removeClass('hide-all');
          $('.js-mincart-error').hide().addClass('hide-all');
        } else {
          $('.js-mincart-error').show().removeClass('hide-all');
          $(Global_Store.checkoutForm).hide().addClass('hide-all');
        }
        // console.log(data);
        if(data.productId == productId && data.quantity > 0)
          Global_Store.AddProduct();
        // $('.js-total-amount').html(data.totalPrice);
        // $('.js-cart-value').text(data.quantity);
  
        // console.log('data.quantity ' + data.quantity);
        // console.log('data.totalPrice ' + data.totalPrice);
  
        $('.js-ptotal-amount').html(data.totalPrice);
        $('.js-pcart-value').text(data.quantity);
        $('.js-pcart-quantity').each(function() { 
          if($(this).parent().find('.cart-label').length !== 0){
            $(this).text('('+data.quantity+')');
          } else{
            $(this).text(data.quantity);
          }
        })
  
        if( data.quantity == 0) {
          $('.js-floating-pcart-empty').show();
          $('.js-checkout-container').hide();
          // prince code added
          $('.js-floating-cart').addClass('cart-not-active');
        }
  
        if( data.quantity == 0) {
          $('.js-floating-pcart-empty').show();
          $(Global_Store.floatingCartBtn).addClass('cart-not-active');
          $('.js-pcheckout-container').hide();
          $('.js-cart-qty-container').hide();
        } else {
          $('.js-floating-pcart-empty').hide();
          $('.js-pcheckout-container').show();
          $(Global_Store.floatingCartBtn).removeClass('cart-not-active');
          $('.js-proofing-header-cart').addClass('favourite-count-active');
          $('.js-cart-qty-container').show();
        }
  
        if( data.quantity > 1)
          $('.js-cart-value-label').text($('.js-itemlabel-holder').data('itemslabel'));
        else
          $('.js-cart-value-label').text($('.js-itemlabel-holder').data('itemlabel'));
          $(temp).find('.js-cart-item-remove').attr('data-itemremove', itemremove);
  
        $.each($('.js-store-cart-item-listing .js-cart-item-list'), function( key, value ) {
          if( data.productId == $(value).data('productid') && data.optionId == $(value).data('optionlabel')) {
            $(value).find('.js-inside-cart-quantity').attr('data-itemid', data.itemId);
            $(value).find('.js-store-add-cart-quantity').attr('data-itemid', data.itemId);
            $(value).find('.js-cart-item-remove').attr('data-itemid', data.itemId);          
            // console.log("datait---" + data.itemId);
          }
        });
  
        $that.val($that.data('label'));
  
      });
  
      Global_Store.AddProduct = function() {
        let variantTitleArray = typeof productVariationData !== 'undefined' ? productVariationData.selectedVariant.variants : [];
        temp = $("#cart-item-template").html();
        temp = $(temp)[0];
  
        $(temp).find('.crop-image-view').removeClass('jcrop-img-view');
        $(temp).find('.crop-image').removeClass('js-crop-image');
        $(temp).find('.crop-image').hide();
        // console.log(temp);
        $(temp).attr('data-optionlabel', labelId);
        $(temp).attr('data-productid', productId);
        // console.log("image " + image + " title " + title + " quantity " + quantity + " price " + price);
        // console.log("imaeg " + image);
        // console.log($('.js-cart-item-list');
        $(temp).find('.js-cart-item-remove').attr('data-product_type', 'store');
        $(temp).find('.cart-item-image').attr('src', image);
        $(temp).find('.cart-item-title').html(title);
        //For Variant Label
        if (variantTitleArray != undefined && variantTitleArray.length > 0) {
          let tempLabel = ''
          variantTitleArray.forEach(function (item) {
            tempLabel += item.product_variation.title + ': ' + item.title + ' </br>';
          });
          $(temp).find('.cart-item-label').html(tempLabel);
        } else {
          $(temp).find('.cart-item-label').html(labelValue);
        }
  
        $(temp).find('.cart-item-unitprice').html(currency+price);
        $(temp).find('.js-inside-cart-quantity').attr('data-itemid', productId);
        $(temp).find('.js-inside-cart-quantity').attr('data-product_type', 'store');
        $(temp).find('.cart-item-quantity').val(quantity);
        if(quantity_threshold){
          // $(temp).find('.cart-item-quantity, .product_quantity_container').css('visibility', 'hidden');
          $(temp).find('.cart-item-quantity, .product_quantity_container').css('display', 'none');
          $(temp).find('.product_quantity_container').hide();      
          $(temp).find('.js-cart-item-remove').addClass('margin-l0');
          $(temp).find('.js-cart-item-remove').removeClass('menu-icon');
          $(temp).find('.js-cart-item-remove').text('Remove');
          
        }
        // $(temp).find('.cart-item-quantity').attr('data-prevvalue', quantity);
        // $(temp).find('.cart-item-quantity').attr('data-title', title);
        // $(temp).find('.cart-item-quantity').attr('data-maxlimit', $('.js-quantity').attr('max'));
  
        $(temp).find('.cart-item-quantity').attr({
          'data-maxlimit': $('.js-quantity').attr('max'),
          'data-title': title,
          'data-prevvalue': quantity,
        });
  
        $(temp).find('.js-store-add-cart-quantity').attr({
          'data-value': quantity,
          'data-title': title,        
          'data-maxlimit': $('.js-quantity').attr('max'),
          'data-prevvalue': quantity,
          'data-product_type': 'store'
        });
        $(temp).find('.cart-item-price').html(currency+''+parseFloat(price*quantity).toFixed(2)+"");
        if($('.js-store-cart-item-listing .js-cart-item-list').length != 0) {
          $.each($('.js-store-cart-item-listing .js-cart-item-list'), function( key, value ) {
            if( $(value).data('productid') == productId && $(value).data('optionlabel') == labelId) {
              $(value).remove();
              counter = 1;
            } else {
              counter = 1;
            }
  
          });
  
        } else {
          $('.js-pcart-item-container').find('.js-store-cart-item-listing').append(temp);
        }
  
        if(counter == 1)
          $('.js-pcart-item-container').find('.js-store-cart-item-listing').append(temp);
  
        var n = noty($.extend({
          text        : itemadded,
          type        : 'success',
          template    : '<div class="noty_message text-left left">'+appendfavimg+notytitle+'<span class="noty_text before-none"></span><div class="noty_close"></div></div>',
          timeout     :  6000,
        }, window.notyDefaults));
      }
  
    });
  
    $(Global_Store.listingAddCartBtn).on('click', function(event) {
      $that = $(this);
      productId = $(this).data('pid');
      image = $(this).data('img');
      title = $(this).data('title');
      labelId = $(this).parents('.js-product-listing-details').find('.js-listing-option-label').val();
      quantity_threshold = $(this).data('quantity_threshold');
      variantTitles = $(this).attr('data-variant_titles');
      // console.log(productId);
      quantity = $('.js-quantity').val();
      appendfavimg = '<div class="js-favimg-noty favimg-noty left"><img src="'+image+'"></div>';
      notytitle = '<div class="js-favimg-title noty-favimg-title">'+title+'</div>';
  
      // console.log( Number.isInteger(parseFloat(quantity)) );
      if( quantity <= 0) {
        sweetAlert((typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_zero : "Oops..."), (typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_cannot_zero : "Quantity cannot be zero!"), "error");
        return false;
      } else if( parseInt(quantity) > parseInt($('.js-quantity').attr('max')) ) {
        sweetAlert("Oops...", "You can add only "+ parseInt($('.js-quantity').attr('max'))+' '+title+ " to the Cart!", "error");
        return false;
      } else if( !Number.isInteger(parseFloat(quantity)) ) {
        // console.log(quantity);
        quantity = 1;
        $('.js-quantity').val(1);
      }
  
  
      $('.js-floating-pcart-empty').hide();
      $('.js-checkout-container').show();
      $('.js-floating-pcart').removeClass('cart-not-active');
  
      if(labelId == undefined)
        labelId = 0;
  
      if(labelId)
        price = $(this).parents('.js-product-listing-details').find('.js-listing-option-label option:selected').attr('data-price');
      else
        price = $(this).data('baseprice');
  
      currency = $('.js-global-currency').val();
      labelValue = $(this).parents('.js-product-listing-details').find('.js-listing-option-label option:selected').attr('data-label');
      counter = 0;
  
      $(this).val('ADDING...');
  
      $.ajax({
          type: 'POST',
          // url: '/cart/' + productId +'?_token='+ csrf_token,
          url: '/cart?_token='+ csrf_token,
          data: {
            'option': labelId,
            'quantity': quantity,
            'productid':productId,
            'global' : '1',
          }
        })
      .done(function(data) {
        // console.log('Response came!')
        // console.log(data);
        if( parseInt(data.totalPrice) >= parseInt($('.js-price-list-body').attr('data-mincartprice')) ) {
          $(Global_Store.checkoutForm).show().removeClass('hide-all');
          $('.js-mincart-error').hide().addClass('hide-all');
        } else {
          $('.js-mincart-error').show().removeClass('hide-all');
          $(Global_Store.checkoutForm).hide().addClass('hide-all');
        }
  
        if(data.productId == productId && data.quantity > 0 )
          Global_Store.AddItem();
        $('.js-ptotal-amount').html(data.totalPrice);
        $('.js-pcart-value').text(data.quantity);
        $('.js-pcart-quantity').each(function() {
          if($(this).parent().find('.cart-label').length !== 0){
            $(this).text('('+data.quantity+')');
          } else{
            $(this).text(data.quantity);
          }
        })
  
        // proofing code added start
        if( data.quantity == 0) {
          // $('.js-floating-cart-empty').show();
          // $('.js-checkout-container').hide();
  
          $('.js-floating-pcart-empty').show();
          $('.js-checkout-container').hide();
  
          // prince code added
          $('.js-floating-cart').addClass('cart-not-active');
        }
  
        if( data.quantity == 0) {
          // $('.js-floating-cart-empty').show();
          // $('.js-checkout-container').hide();
          // $('.js-floating-cart').addClass('cart-not-active');
  
          $('.js-floating-pcart-empty').show();
          $(Global_Store.floatingCartBtn).addClass('cart-not-active');
          $('.js-pcheckout-container').hide();
  
          $('.js-cart-qty-container').hide();
        } else {
          $('.js-floating-pcart-empty').hide();
          $('.js-pcheckout-container').show();
          $(Global_Store.floatingCartBtn).removeClass('cart-not-active');
          $('.js-proofing-header-cart').addClass('favourite-count-active');
  
          $('.js-cart-qty-container').show();
        }
        // proofing code added end
  
        if( data.quantity > 1)
          $('.js-cart-value-label').text($('.js-itemlabel-holder').data('itemslabel'));
        else
          $('.js-cart-value-label').text($('.js-itemlabel-holder').data('itemlabel'));
          $(temp).find('.js-cart-item-remove').attr('data-itemremove', itemremove);
  
        $.each($('.js-store-cart-item-listing .js-cart-item-list'), function( key, value ) {
          if( data.productId == $(value).data('productid') && data.optionId == $(value).data('optionlabel')) {
            $(value).find('.js-inside-cart-quantity').attr('data-itemid', data.itemId);
            $(value).find('.js-cart-item-remove').attr('data-itemid', data.itemId);
            $(value).find('.js-store-add-cart-quantity').attr('data-itemid', data.itemId);
          }
        });
  
        $that.val($that.data('label'));
  
      });
  
      Global_Store.AddItem = function() {
        temp = $("#cart-item-template").html();
        temp = $(temp)[0];
  
        $(temp).find('.crop-image-view').removeClass('jcrop-img-view');
        $(temp).find('.crop-image').removeClass('js-crop-image');
        $(temp).find('.crop-image').hide();
        // console.log(temp);
  
        $(temp).attr('data-optionlabel', labelId);
        $(temp).attr('data-productid', productId);
  
        // $(temp).children('.js-inside-cart-quantity').attr('data-itemid', productId);
        $(temp).find('.js-inside-cart-quantity').attr('data-itemid', productId);
        $(temp).find('.js-inside-cart-quantity').attr('data-product_type', 'store');
  
        $(temp).find('.js-cart-item-remove').attr('data-product_type', 'store');
  
  
        $(temp).find('.cart-item-image').attr('src', image);
        $(temp).find('.cart-item-title').html(title);
        //For Variant Label
        if (variantTitles != undefined && variantTitles != '') {
          let tempLabel = '';
          $.each(JSON.parse(variantTitles), function (parentTitle, varTitle) {
            tempLabel += parentTitle + ': ' + varTitle + ' </br>';
          });
          $(temp).find('.cart-item-label').html(tempLabel);
        } else {
          $(temp).find('.cart-item-label').html(labelValue);
        }
  
        $(temp).find('.cart-item-unitprice').html(currency+price);
        $(temp).find('.cart-item-quantity').val(quantity);
        // console.log('dsadsad'+quantity_threshold);
  
        if(quantity_threshold){
          $(temp).find('.cart-item-quantity').css('visibility', 'hidden');
        }
        $(temp).find('.cart-item-quantity').attr('data-prevvalue', quantity);
        $(temp).find('.cart-item-quantity').attr('data-title', title);
        $(temp).find('.cart-item-quantity').attr('data-maxlimit', $('.js-quantity').attr('max'));
        $(temp).find('.cart-item-price').html(currency+''+parseFloat(price*quantity).toFixed(2)+"");
  
        $(temp).find('.js-store-add-cart-quantity').attr({
          'data-value': quantity,
          'data-title': title,
          'data-maxlimit': $('.js-quantity').attr('max'),
          'data-prevvalue': quantity
        });
        
        if($('.js-store-cart-item-listing .js-cart-item-list').length != 0) {
          $.each($('.js-store-cart-item-listing .js-cart-item-list'), function( key, value ) {
            if( $(value).data('productid') == productId && $(value).data('optionlabel') == labelId) {
              $(value).remove();
              counter = 1;
            } else {
              counter = 1;
            }
          });
        } else {
          $('.js-pcart-item-container .js-store-cart-item-listing').append(temp);
        }
  
        if(counter == 1)
          $('.js-pcart-item-container .js-store-cart-item-listing').append(temp);
  
        var n = noty($.extend({
          text        : $('.js-store-cart-item-listing').data('itemadded'),
          type        : 'success',
          template    : '<div class="noty_message text-left left">'+appendfavimg+notytitle+'<span class="noty_text before-none"></span><div class="noty_close"></div></div>',
          timeout     :  6000,
        }, window.notyDefaults));
      }
      //Store.AddItem();
    });
    
    // js-storeproduct-add-cart-quantity
    $(Global_Store.productDetailAddCartBtn).unbind('click');
    $(Global_Store.productDetailAddCartBtn).on('click', function(event) {
      var quan = parseInt($(this).siblings('input').val());
      spanCount = parseInt($(this).siblings('input').val());
      maxLimitQty = $(this).attr('data-maxlimit');
      $that = $(this);
      if( $(this).hasClass('plus-icon') ) {
        quan = spanCount+1;
        if( parseInt(quan) > parseInt($(this).attr('data-maxlimit')) ) {
          sweetAlert("Oops...", "You can add only "+ parseInt($(this).attr('data-maxlimit'))+' '+$(this).attr('data-title')+ " to the Cart!", "error");
          return false;
        }
        $(this).siblings('input').val(spanCount+1);
      } else {
        if( (spanCount-1) <= 0 ) {
          sweetAlert((typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_zero : "Oops..."), (typeof window.labels != 'undefined' ? window.labels.store_cart_item_quantity_cannot_zero : "Quantity cannot be zero!"), "error");
          return false;
        } else if( !Number.isInteger(parseFloat(quan)) ) {
          $(this).siblings('input').val(spanCount-1);
          quan = spanCount-1;
        } else{
          $(this).siblings('input').val(spanCount-1);
          quan = spanCount-1;
        }
      }
  
      // svg addClassname added
      var svgIcons = $(this).find('svg use').attr('xlink:href');
      $(this).css('pointer-events', 'none');
      $(this).find('svg use').attr('xlink:href', '');
      $(this).find('svg use').attr('xlink:href', '#cart-loading-pixpa-icon');
  
      if (quan == 1) {
        if( $(this).hasClass('fa-minus') ) {
          $(this).addClass('active')
        }
      } else{
        $(this).siblings('.fa-minus').removeClass('active')
      }
  
      setTimeout(function() {
        $that.css('pointer-events', '');
        $that.find('svg use').attr('xlink:href', svgIcons);
      }, 300);
    })
  
  }
  
  _JCROP = {};
  _JCROP.init = function() {
    // console.log('Initialize jcrop!');
    function showCoords(c) {
      _JCROP.coor = c;
      getPoints();
    }
  
    jQuery(function($) {
      $('.magic').Jcrop({
        onSelect : showCoords,
        onChange : showCoords,
        setSelect : [_JCROP.boxData.x, _JCROP.boxData.y, _JCROP.boxData.w, _JCROP.boxData.h],
        allowResize : false,
        allowSelect : false,
        boxWidth: 500,   //Maximum width you want for your bigger images
        boxHeight: 550,  //Maximum Height for your bigger images
      }, function() {
        _JCROP.jcrop = this;
      });
    });
  }
  
  
  
  Global_Store.crop_init = function (th) {
      _JCROP.boxData = {};
      _JCROP.active = th;
      _JCROP.boxData.x = $(th).attr('data-x');
      _JCROP.boxData.y = $(th).attr('data-y');
      _JCROP.boxData.h = $(th).attr('data-h');
      _JCROP.boxData.w = $(th).attr('data-w');
      _JCROP.asp = { x : parseFloat($(th).attr('data-aspx')), y : parseFloat($(th).attr('data-aspy')) };
  
      _JCROP.cartid = $(th).parent().parent().find(Global_Store.insideCartQuantity).attr('data-itemid');
  
      var i = new Image();
      i.className = "magic";
      $('.crop-image-wrapper').html(i);
  
      // if (userObject.isMobile == '1') {
      //   // var src = get500Image($(th).parent().parent().find('img').attr('src'));
      // } else {
      //   // var src = get500Image($(th).parent().parent().parent().find('img').attr('src'));
      // }
  
      var src = get500Image($(th).attr('data-src'));
  
      i.onload = function() {
        _JCROP.img = {};
        _JCROP.img.el = i;
        _JCROP.img.x = i.width;
        _JCROP.img.y = i.height;
        _JCROP.init();
      }
      i.src = src;
  
  }
  
  
  Global_Store.cropModalInit = function() {
    $(Global_Store.cropImageBtn).unbind('click');
    $(Global_Store.cropImageBtn).on('click', function(event) {
  
      $(Global_Store.cropImage).addClass('image-crop-modal-active bg-white js-crop-modal-bg');
      $(".js-image-crop-body-overlay, .js-image-crop-body").addClass('image-crop-modal-active js-crop-modal-body');
      Global_Store.crop_init(this);
  
      $('body').find('.jcrop-img-view').removeClass('js-update_jcrop');
      $('body').find('.js-cart-item-list').removeClass('js-update_jcrop_bg');
      $(this).parent().parent().siblings('.jcrop-img-view').addClass('js-update_jcrop');
      $(this).parent().parent().parent('.js-cart-item-list').addClass('js-update_jcrop_bg');
  
    });
  
    $(Global_Store.cropImageCloseBtn).unbind('click');
    $(Global_Store.cropImageCloseBtn).on('click', function() {
      $(Global_Store.cropImage).removeClass('image-crop-modal-active bg-white js-crop-modal-bg');
      $(".js-image-crop-body-overlay, .js-image-crop-body").removeClass('image-crop-modal-active');
      if($('.magic').length>0) {
        $('.magic').remove();
      }
      if(typeof _JCROP.jcrop != 'undefined') {
        _JCROP.jcrop.destroy();
        // console.log('Jcrop destroyed!');
      }
  
      $('body').find('.jcrop-img-view').removeClass('js-update_jcrop');
      $('body').find('.js-cart-item-list').removeClass('js-update_jcrop_bg');
  
    });
  
    $(Global_Store.cropCartUpdate).unbind('click');
    $(Global_Store.cropCartUpdate).on('click', function() {
      updateCartCoordinates();
  
      $(function(){
        Global_Store.jCropintUpdate();
      });
  
    });
  
    $(".js-rotate-crop").unbind('click');
    $(".js-rotate-crop").on('click', function() {
      Global_Store.rotateCrop();
    });
  }
  
  function get500Image (imgsrc) {
    // if (userObject.isMobile == '1') {
    //   return imgsrc.replace('/100/', '/500/');
    // } else{
      // return imgsrc.replace('/100/', '/500/');
      return imgsrc;
    // }
  }
  
  function getP(x,i) {
    return Math.round((x*100)/i);
  }
  
  function getPoints() {
    _JCROP.whcc = {};
    var xcenter = Math.round((_JCROP.coor.x+_JCROP.coor.x2)/2);
    var ycenter = Math.round((_JCROP.coor.y+_JCROP.coor.y2)/2);
    _JCROP.whcc.x1 = getP(xcenter, _JCROP.img.x);
    _JCROP.whcc.y1 = getP(ycenter, _JCROP.img.y);
    _JCROP.whcc.x3 = getP(_JCROP.coor.w, _JCROP.img.x);
    _JCROP.whcc.y3 = getP(_JCROP.coor.h, _JCROP.img.y);
    _JCROP.whcc.x2 = 50;
    _JCROP.whcc.y2 = 50;
  }
  
  function updateCartCoordinates() {
    if(!_JCROP.cartid || typeof _JCROP.cartid == 'undefined') {
      alert("Item is no longer available in your cart.");
      return false;
    }
    var cartid = _JCROP.cartid;
    var route_arr = $('.js-price-list-body').data('requestslug').split('/');
    route_arr[3] = "update-crop";
    var update_route = route_arr.join('/');
  
    $.ajax({
      type: 'POST',
      url: update_route+'?_token='+$('#csrf_token').val(),
      data: {
        'cartid': cartid,
        'data_x': _JCROP.coor.x,
        'data_y': _JCROP.coor.y,
        'data_h': parseInt(_JCROP.coor.y + _JCROP.coor.h),
        'data_w': parseInt(_JCROP.coor.x + _JCROP.coor.w),
        'data_oh': parseInt(_JCROP.coor.h),
        'data_ow': parseInt(_JCROP.coor.w),
        'print_width': parseFloat(_JCROP.asp.x),
        'print_height': parseFloat(_JCROP.asp.y),
      }
    })
    .done(function(data) {
      //update data-coordinates
      _JCROP.active.setAttribute('data-x', _JCROP.coor.x);
      _JCROP.active.setAttribute('data-y', _JCROP.coor.y);
      _JCROP.active.setAttribute('data-h', parseInt(_JCROP.coor.y + _JCROP.coor.h));
      _JCROP.active.setAttribute('data-w', parseInt(_JCROP.coor.x + _JCROP.coor.w));
      _JCROP.active.setAttribute('data-oh', parseInt(_JCROP.coor.h));
      _JCROP.active.setAttribute('data-ow', parseInt(_JCROP.coor.w));
      _JCROP.active.setAttribute('data-aspx', parseFloat(_JCROP.asp.x));
      _JCROP.active.setAttribute('data-aspy', parseFloat(_JCROP.asp.y));
  
      _JCROP.active.parentElement.parentElement.previousElementSibling.setAttribute('data-x', _JCROP.coor.x);
      _JCROP.active.parentElement.parentElement.previousElementSibling.setAttribute('data-y', _JCROP.coor.y);
      _JCROP.active.parentElement.parentElement.previousElementSibling.setAttribute('data-h', parseInt(_JCROP.coor.y + _JCROP.coor.h));
      _JCROP.active.parentElement.parentElement.previousElementSibling.setAttribute('data-w', parseInt(_JCROP.coor.x + _JCROP.coor.w));
      _JCROP.active.parentElement.parentElement.previousElementSibling.setAttribute('data-oh', parseInt(_JCROP.coor.h));
      _JCROP.active.parentElement.parentElement.previousElementSibling.setAttribute('data-ow', parseInt(_JCROP.coor.w));
      _JCROP.active.parentElement.parentElement.previousElementSibling.setAttribute('data-aspx', parseFloat(_JCROP.asp.x));
      _JCROP.active.parentElement.parentElement.previousElementSibling.setAttribute('data-aspy', parseFloat(_JCROP.asp.y));
  
      // sweetAlert('Cropped Image coordinates updated');
      // swal({
      //   title: "Crop area has been updated",
      //   text: "",
      //   customClass: 'image-crop-alert',
      //   confirmButtonText: "OK",
      //   allowOutsideClick: false,
      // });
  
      var n = noty($.extend({
        text        : window.labels != 'undefined' ? window.labels.crop_area_has_been_updated : 'Crop area has been updated',
        type        : 'success',
        template    : '<div class="noty_message"><span class="noty_text"></span><div class="noty_close"></div></div>',
        timeout     :  4000,
      }, window.notyDefaults));
  
  
  
    });
  }
  
  
  
  Global_Store.jCropint = function() {
    $(".jcrop-img-view img").each(function() {
      var x = $(this).parents('.jcrop-img-view').attr('data-x');
      var y = $(this).parents('.jcrop-img-view').attr('data-y');
      var h = $(this).parents('.jcrop-img-view').attr('data-h');
      var w = $(this).parents('.jcrop-img-view').attr('data-w');
      var aspx = parseFloat($(this).parents('.jcrop-img-view').attr('data-aspx'));
      var aspy = parseFloat($(this).parents('.jcrop-img-view').attr('data-aspy'));
      // console.log(x, y, h, w, aspx / aspy);
      $(this).Jcrop({
        // onSelect:    showCoords,
        bgColor:     'black',
        bgOpacity:   .4,
        setSelect:   [ x, y, h, w ],
        aspectRatio: aspx / aspy,
        allowResize : false,
        allowSelect : false,
        boxWidth: 70, boxHeight: 105,
        // boxWidth: w, boxHeight: h,
        trueSize: [w,h]
      });
      $(this).parents('.jcrop-img-view').addClass('jcropadded');
      // console.log('princexxx img src - ', $(this).attr('src') );
    });
  };
  
  Global_Store.jCropAppendint = function(jcropImg) {
    // $(".jcrop-img-view img").each(function() {
      var x = $(jcropImg).parents('.jcrop-img-view').attr('data-x');
      var y = $(jcropImg).parents('.jcrop-img-view').attr('data-y');
      var h = $(jcropImg).parents('.jcrop-img-view').attr('data-h');
      var w = $(jcropImg).parents('.jcrop-img-view').attr('data-w');
      var aspx = parseFloat($(jcropImg).parents('.jcrop-img-view').attr('data-aspx'));
      var aspy = parseFloat($(jcropImg).parents('.jcrop-img-view').attr('data-aspy'));
      // console.log('princexxx append - ', x, y, h, w, aspx / aspy);
      $(jcropImg).Jcrop({
        bgColor:     'black',
        bgOpacity:   .4,
        setSelect:   [ x, y, h, w ],
        aspectRatio: aspx / aspy,
        allowResize : false,
        allowSelect : false,
        boxWidth: 70, boxHeight: 105,
        trueSize: [w,h]
      });
    // });
  };
  
  
  Global_Store.jCropintUpdate = function() {
    setTimeout( function (){
      var x = $('.js-update_jcrop').attr('data-x');
      var y = $('.js-update_jcrop').attr('data-y');
      var h = $('.js-update_jcrop').attr('data-h');
      var w = $('.js-update_jcrop').attr('data-w');
      var aspx = parseFloat($('.js-update_jcrop').attr('data-aspx'));
      var aspy = parseFloat($('.js-update_jcrop').attr('data-aspy'));
      // console.log(x, y, h, w, aspx / aspy);
      // console.log("img icons- update crop");
      $('.js-update_jcrop img').Jcrop({
        // onSelect:    showCoords,
        bgColor:     'black',
        bgOpacity:   .4,
        setSelect:   [ x, y, h, w ],
        aspectRatio: aspx / aspy,
        allowResize : false,
        allowSelect : false,
        boxWidth: 70, boxHeight: 105,
        trueSize: [w,h]
      });
      // console.log("img icons- update crop done");
    },1000);
  };
  
  