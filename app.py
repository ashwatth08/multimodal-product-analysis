import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
import torch.nn.functional as F

st.set_page_config(
    page_title="Universal Multimodal Product Defect & Sentiment Auditor",
    page_icon="🔍",
    layout="wide"
)

# 1. Load Pretrained Models
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    text_sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    return clip_model, clip_proc, text_sentiment, device

clip_model, clip_proc, text_pipe, device = load_models()

# 2. Taxonomy & Specialized Defect Prompts
PRODUCT_TAXONOMY = {
    "Footwear & Shoes": {
        "detect_prompt": "a pair of shoes, sneakers, boots, or footwear",
        "defect_prompt": "heavily worn out, dirty, scuffed, damaged shoes with sole separation, creasing, peeling leather, or holes",
        "intact_prompt": "a clean, brand new pair of sneakers or shoes in crisp pristine unworn condition",
        "defect_title": "Footwear Wear & Material Degradation"
    },
    "Clothing & Apparel": {
        "detect_prompt": "clothing, apparel, fabric, jacket, shirt, pants, or bag",
        "defect_prompt": "torn fabric, broken zipper, frayed seams, stains, missing buttons, or damaged apparel",
        "intact_prompt": "brand new, flawless clothing or apparel without any tears, stains, or damage",
        "defect_title": "Apparel & Fabric Flaw / Wear"
    },
    "Electronics & Displays": {
        "detect_prompt": "an electronic device, smartphone, laptop, screen, or headphones",
        "defect_prompt": "defective screen with green lines, shattered glass, cracked casing, torn ear cushions, or hardware damage",
        "intact_prompt": "a brand new, flawless electronic device with an intact display screen and no scratches or cracks",
        "defect_title": "Screen & Hardware Defect"
    },
    "Kitchenware & Appliances": {
        "detect_prompt": "a kitchen appliance, cookware, toaster, blender, kettle, or ceramic dish",
        "defect_prompt": "cracked plastic casing, chipped ceramic, broken glass jar, burnt heating element, or dented metal",
        "intact_prompt": "a brand new, flawless kitchen appliance or cookware in pristine condition",
        "defect_title": "Kitchenware Fracture & Casing Damage"
    },
    "Furniture & Home Decor": {
        "detect_prompt": "furniture, chair, table, mirror, wooden or metal home decor",
        "defect_prompt": "scratched wood, broken chair leg, chipped paint, bent frame, or shattered glass mirror",
        "intact_prompt": "brand new, polished, sturdy furniture in flawless condition",
        "defect_title": "Structural & Surface Damage"
    },
    "Shipping & Packaging": {
        "detect_prompt": "cardboard box, parcel packaging, shipping carton, or sealed bottle",
        "defect_prompt": "crushed shipping box, torn parcel, leaked liquid bottle, punctured packaging, or broken seal",
        "intact_prompt": "an intact, undamaged, clean, factory-sealed delivery box or packaging",
        "defect_title": "Transit Packaging & Seal Damage"
    }
}

st.title("🔍 Universal Multimodal Product Defect & Sentiment Auditor")
st.markdown("Context-aware auditing across **Electronics, Footwear, Fashion, Appliances, Furniture, and Packaging**.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Submission Data")
    uploaded_image = st.file_uploader("Upload Inspection / Delivery Photo", type=["jpg", "jpeg", "png"])
    review_text = st.text_area(
        "Customer Review Text",
        value="Not good quality, used product",
        height=130
    )
    analyze_btn = st.button("Run Audit", type="primary", use_container_width=True)

with col_right:
    st.subheader("2. Inspection Results")
    if uploaded_image:
        raw_img = Image.open(uploaded_image).convert("RGB")
        st.image(raw_img, caption="Inspected Item", width=280)
    else:
        raw_img = None
        st.info("Upload a product photo to begin inspection.")

    if analyze_btn:
        if not review_text.strip():
            st.warning("Please enter review text.")
        elif raw_img is None:
            st.warning("Please upload an image for multimodal verification.")
        else:
            with st.spinner("Classifying category and auditing condition..."):
                # --- Step A: Text Sentiment ---
                txt_res = text_pipe(review_text)[0]
                txt_sentiment = "Positive" if txt_res["label"] == "POSITIVE" else "Negative"
                txt_score = txt_res["score"]

                # --- Step B: Auto-Detect Product Category ---
                categories = list(PRODUCT_TAXONOMY.keys())
                cat_prompts = [PRODUCT_TAXONOMY[cat]["detect_prompt"] for cat in categories]
                
                cat_inputs = clip_proc(text=cat_prompts, images=raw_img, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    cat_probs = clip_model(**cat_inputs).logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()
                
                detected_cat = categories[cat_probs.argmax()]
                cat_config = PRODUCT_TAXONOMY[detected_cat]

                # --- Step C: Category-Specific Visual Inspection ---
                inspection_prompts = [
                    cat_config["defect_prompt"],
                    cat_config["intact_prompt"]
                ]
                inspect_inputs = clip_proc(text=inspection_prompts, images=raw_img, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    inspect_probs = clip_model(**inspect_inputs).logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()

                p_defect = float(inspect_probs[0])
                p_intact = float(inspect_probs[1])

                # --- Step D: Strict Evaluation Logic ---
                is_damaged = (p_defect > 0.50)
                
                if is_damaged:
                    img_condition = "Defective / Damaged / Used"
                    visual_confidence = p_defect * 100
                else:
                    img_condition = "Intact / Good Condition"
                    visual_confidence = p_intact * 100

                # --- Step E: Output Diagnostics ---
                st.write(f"### 📦 Detected Category: **{detected_cat}**")
                m1, m2 = st.columns(2)
                m1.metric("Visual Condition", img_condition, f"{visual_confidence:.1f}% confidence")
                m2.metric("Customer Stated Tone", txt_sentiment, f"{txt_score*100:.1f}% confidence")

                st.write("---")

                # Decision Matrix
                if txt_sentiment == "Negative" and is_damaged:
                    st.error("🚨 **CONFIRMED DEFECT / LEGITIMATE COMPLAINT**")
                    st.markdown(
                        f"""
                        * **Category:** {detected_cat}
                        * **Customer Review:** Negative complaint verified (*"{review_text}"*).
                        * **Visual Verification:** {cat_config['defect_title']} confirmed (**{visual_confidence:.1f}% confidence**).
                        * **Verdict:** Verified legitimate defect. Approved for replacement or return.
                        """
                    )
                elif txt_sentiment == "Positive" and is_damaged:
                    st.error("🚨 **CRITICAL CONTRADICTION DETECTED (Sarcastic Review / Hidden Defect)**")
                    st.markdown(
                        f"""
                        * **Category:** {detected_cat}
                        * **Customer Review:** Positive review (*"{review_text}"*).
                        * **Visual Verification:** Visual inspection detected physical flaws (**{visual_confidence:.1f}% confidence**).
                        * **Verdict:** Flagged for quality inspection audit (text contradicts visual damage).
                        """
                    )
                elif txt_sentiment == "Negative" and not is_damaged:
                    st.warning("⚠️ **POTENTIAL MISMATCH / UNSUBSTANTIATED COMPLAINT**")
                    st.markdown(
                        f"""
                        * **Category:** {detected_cat}
                        * **Customer Review:** Customer stated negative remarks (*"{review_text}"*).
                        * **Visual Verification:** Product photo appears clean, intact, and in good condition (**{visual_confidence:.1f}% confidence**).
                        * **Verdict:** Unsubstantiated return claim. Routed for manual customer support audit.
                        """
                    )
                else:
                    st.success(f"✅ **PRODUCT VERIFIED: INTACT & SATISFACTORY ({detected_cat})**")

                # Dynamic Signals Display
                st.write("#### Detailed Visual Signal Scores")
                st.write(f"**{cat_config['defect_title']} Probability:** {p_defect*100:.1f}%")
                st.progress(min(p_defect, 1.0))
                st.write(f"**Intact / Pristine Condition Probability:** {p_intact*100:.1f}%")
                st.progress(min(p_intact, 1.0))