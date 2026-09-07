import argparse
import os
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing import image as keras_image # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt

CLASS_NAMES   = ['grade A', 'grade B', 'grade C']
CLASS_COLORS  = {'grade A':"limegreen", 'grade B': "orange", 'grade C': 'red'}
CLASS_ADVICE  = {
    'grade A':    'Execellent wheat quality.',
    'grade B': 'Good wheat quality.',
    'grade C':     'acceptable Quality.'
}
IMG_SIZE      = (224, 224)
MODEL_PATH    = "wheat_classifier_mobilenetv2.keras"
TTA_ROUNDS    = 8


def load_model(path=MODEL_PATH):
    print(f"Loading model from: {path}")
    model = tf.keras.models.load_model(path)
    print("Model loaded successfully.\n")
    return model


def load_and_preprocess(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img)
    arr_pre = preprocess_input(arr.copy())
    return arr_pre, img, arr


def predict_single(model, img_path):
    arr_pre, original_img, _ = load_and_preprocess(img_path)
    probs = model.predict(np.expand_dims(arr_pre, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs, original_img


def predict_tta(model, img_path, n_rounds=TTA_ROUNDS):
    _, original_img, arr = load_and_preprocess(img_path)
    augmenter = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        width_shift_range=0.05,
        height_shift_range=0.05
    )
    preds = []
    for _ in range(n_rounds):
        aug_arr = augmenter.random_transform(arr.copy())
        aug_pre = preprocess_input(aug_arr)
        p = model.predict(np.expand_dims(aug_pre, 0), verbose=0)[0]
        preds.append(p)

    avg_probs = np.mean(preds, axis=0)
    idx = int(np.argmax(avg_probs))
    return CLASS_NAMES[idx], float(avg_probs[idx]) * 100, avg_probs, original_img


def predict(model, img_path, use_tta=False):
    return predict_tta(model, img_path) if use_tta else predict_single(model, img_path)


def display_result(img_path, class_name, confidence, probs, use_tta=False):
    _, original_img, _ = load_and_preprocess(img_path)
    color = CLASS_COLORS[class_name]
    mode = "TTA" if use_tta else "Single Pass"

   
    fig = plt.figure(figsize=(6, 7), facecolor='white')

    # Reserve top 78% for the image, bottom 22% for the text block
    ax_img  = fig.add_axes([0.05, 0.22, 0.90, 0.70])   
    ax_text = fig.add_axes([0.05, 0.00, 0.90, 0.20])

    
    ax_img.imshow(original_img)
    ax_img.axis('off')

    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax_img.spines[spine].set_visible(True)
        ax_img.spines[spine].set_linewidth(5)
        ax_img.spines[spine].set_color(color)

    
    ax_img.set_title(
        'Wheat Grain Quality Analysis',
        fontsize=14, fontweight='bold', pad=12, color='#222222'
    )

    
    ax_text.axis('off')


    ax_text.text(
        0.5, 0.80,
        f"Predicted: {class_name.upper()} ",
        transform=ax_text.transAxes,
        ha='center', va='center',
        fontsize=13, fontweight='bold', color=color
    )

    ax_text.text(
        0.5, 0.50,
        f"Confidence: {confidence:.1f}%",
        transform=ax_text.transAxes,
        ha='center', va='center',
        fontsize=12, color='#333333'
    )
    
    ax_text.text(
        0.5, 0.18,
        CLASS_ADVICE[class_name],
        transform=ax_text.transAxes,
        ha='center', va='center',
        fontsize=10, style='italic', color='#666666'
    )

    
    line = plt.Line2D([0.05, 0.95], [0.225, 0.225],
                      transform=fig.transFigure,
                      color='#dddddd', linewidth=1)
    fig.add_artist(line)

    out_name = os.path.splitext(os.path.basename(img_path))[0] + '_result.png'
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Result saved → {out_name}")


def batch_predict(model, folder_path, use_tta=False):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

    if not files:
        print("No image files found in folder.")
        return

    mode = "TTA" if use_tta else "Single"
    results = {c: 0 for c in CLASS_NAMES}

    print(f"\nClassifying {len(files)} images — mode: {mode}\n")
    print(f"{'File':<38} {'Class':<10} {'Confidence':>12}")
    print("─" * 64)

    for fname in sorted(files):
        fpath = os.path.join(folder_path, fname)
        cls, conf, _, _ = predict(model, fpath, use_tta=use_tta)
        results[cls] += 1
        marker = {'good': '✓', 'average': '~', 'bad': '✗'}[cls]
        print(f"{fname:<38} {marker} {cls:<8} {conf:>10.1f}%")

    total = len(files)
    print("─" * 64)
    print(f"\nSUMMARY ({total} images):")
    for cls, count in results.items():
        pct = count / total * 100
        bar = '█' * int(pct / 5)
        print(f"  {cls:8s}: {count:4d}  ({pct:5.1f}%)  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wheat Grain Quality Classifier')
    parser.add_argument('--model',  default=MODEL_PATH, help='Path to trained model (.keras or .h5)')
    parser.add_argument('--image',  default=None, help='Path to a single wheat image')
    parser.add_argument('--folder', default=None, help='Path to folder of wheat images')
    parser.add_argument('--tta',    action='store_true', help='Enable Test-Time Augmentation')
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image:
        cls, conf, probs, _ = predict(model, args.image, use_tta=args.tta)
        mode = "TTA" if args.tta else "Single pass"
        print(f"Mode   : {mode}")
        print(f"Result : {cls.upper()} ({conf:.1f}% confidence)")
        print(f"Advice : {CLASS_ADVICE[cls]}")
        display_result(args.image, cls, conf, probs, use_tta=args.tta)
    elif args.folder:
        batch_predict(model, args.folder, use_tta=args.tta)
    else:
        print("Please provide --image or --folder argument.")
        