package com.google.ar.core.examples.java.ml.classification

import android.graphics.Color
import com.google.cloud.vision.v1.*
import com.google.protobuf.ByteString
import android.media.Image
import android.util.Log
import com.google.ar.core.examples.java.ml.MainActivity
import com.google.ar.core.examples.java.ml.classification.utils.ImageUtils
import com.google.ar.core.examples.java.ml.classification.utils.ImageUtils.toByteArray
import com.google.ar.core.examples.java.ml.classification.utils.VertexUtils.calculateAverage
import com.google.ar.core.examples.java.ml.classification.utils.VertexUtils.rotateCoordinates
import com.google.ar.core.examples.java.ml.classification.utils.VertexUtils.toAbsoluteCoordinates
import com.google.auth.oauth2.GoogleCredentials
import com.google.cloud.vision.v1.AnnotateImageRequest
import com.google.cloud.vision.v1.Feature
import com.google.cloud.vision.v1.ImageAnnotatorClient
import com.google.cloud.vision.v1.ImageAnnotatorSettings
import com.google.cloud.vision.v1.Image as GCVImage


 class GoogleCloudVisionImage(val activity: MainActivity) : ObjectDetector(activity) {
    companion object {
        val TAG = "GoogleCloudVisionDetector"
    }

    val credentials = try {
        // Providing GCP credentials is not mandatory for this app, so the existence of R.raw.credentials
        // is not guaranteed. Instead, use getIdentifier to determine an optional resource.
        val res = activity.resources.getIdentifier("credentials", "raw", activity.packageName)
        if (res == 0) error("Missing GCP credentials in res/raw/credentials.json.")
        GoogleCredentials.fromStream(activity.resources.openRawResource(res))
    } catch (e: Exception) {
        Log.e(TAG, "Unable to create Google credentials from res/raw/credentials.json. Cloud ML will be disabled.", e)
        null
    }

     val settings = ImageAnnotatorSettings.newBuilder().setCredentialsProvider { credentials }.build()
     val vision = ImageAnnotatorClient.create(settings)

     override suspend fun analyze(image: Image, imageRotation: Int): List<DetectedObjectResult> {
         // Convert the image to a byte array.
         val convertYuv = convertYuv(image)

         // The model performs best on upright images, so rotate it.
         val rotatedImage = ImageUtils.rotateBitmap(convertYuv, imageRotation)

         // Perform request on Google Cloud Vision APIs.
         val request = createAnnotateImageRequest(rotatedImage.toByteArray())
         val response = vision.batchAnnotateImages(listOf(request))

         // Process result and map to DetectedObjectResult.
         val imagePropertiesResult = response.responsesList.first().imagePropertiesAnnotation
         val cropPropertiesResult = response.responsesList.first().cropHintsAnnotation

         //Log.d("COLORS 2", "(${s1}, ${r1}, ${g1}, ${b1}, ${p1})")

         Log.d("COLORS", getMostCol(imagePropertiesResult.dominantColors.colorsList))
         return imagePropertiesResult.dominantColors.colorsList.map {
             val color = it.color
             val r = color.red.toFloat() / 255
             val g = color.green.toFloat() / 255
             val b = color.blue.toFloat() / 255
             val score = it.score
             val pixelFraction = it.pixelFraction
             DetectedObjectResult(score, "(${r}, ${g}, ${b}, ${pixelFraction})",Pair(0,0) )
         }


     }

     private fun getMostCol(ColorArray: List<ColorInfo>): String {
         var mostCon = 0.0f;
         var mostCounter = 0;
         for((index,col) in ColorArray.withIndex())
         {
             if(col.score > mostCon)
             {
                 mostCon = col.score
                 mostCounter = index
             }
             //Log.d("COLORS 2", getColorNameFromRgb(ColorArray.get(index).color.red,ColorArray.get(index).color.green,ColorArray.get(index).color.blue, ColorArray.get(index).score))
         }
         return ColorArray.get(mostCounter).allFields.toString()


     }

     private fun createAnnotateImageRequest(imageBytes: ByteArray): AnnotateImageRequest {
         // GCVImage is a typealias for com.google.cloud.vision's Image, needed to differentiate from android.media.Image
         val image = GCVImage.newBuilder().setContent(ByteString.copyFrom(imageBytes))
         val features = Feature.newBuilder().setType(Feature.Type.IMAGE_PROPERTIES)
         return AnnotateImageRequest.newBuilder()
             .setImage(image)
             .addFeatures(features)
             .build()
     }


 }
