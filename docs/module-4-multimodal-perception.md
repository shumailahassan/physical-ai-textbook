---
id: module-4-multimodal-perception
title: Chapter 2 - Multimodal Perception Systems
sidebar_label: Chapter 2 - Multimodal Perception Systems
---

# Chapter 2: Multimodal Perception Systems

## Visual Perception for Robotics

Visual perception in robotics involves processing visual information to understand the environment and guide robot actions. In VLA systems, visual perception goes beyond simple object detection to encompass scene understanding, spatial reasoning, and visual-language grounding.

### Computer Vision Fundamentals for Robotics

Computer vision in robotics differs from traditional computer vision in several key ways:

- **Real-time processing**: Robotics applications require low-latency processing for safe and responsive behavior
- **3D understanding**: Robots operate in 3D environments, requiring depth and spatial information
- **Embodiment**: Visual processing must account for the robot's physical perspective and capabilities
- **Action relevance**: Visual information must be processed with respect to the robot's potential actions

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

class RobotVisionProcessor:
    """
    Visual perception system for robotics applications
    """
    def __init__(self, model_path=None):
        # Initialize visual perception components
        self.feature_extractor = self._initialize_feature_extractor()
        self.object_detector = self._initialize_object_detector()
        self.segmentation_model = self._initialize_segmentation_model()
        self.depth_estimator = self._initialize_depth_estimator()

        # Preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _initialize_feature_extractor(self):
        # Initialize CNN-based feature extractor
        # Could use ResNet, EfficientNet, or other architectures
        import torchvision.models as models
        model = models.resnet50(pretrained=True)
        # Remove final classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model

    def _initialize_object_detector(self):
        # Initialize object detection model (e.g., YOLO, Faster R-CNN)
        # This is a simplified placeholder
        return ObjectDetectionModel()

    def _initialize_segmentation_model(self):
        # Initialize semantic segmentation model
        # Could use DeepLab, UNet, or similar
        return SegmentationModel()

    def _initialize_depth_estimator(self):
        # Initialize depth estimation model
        # Could use MiDaS, NeRF, or similar
        return DepthEstimationModel()

    def process_frame(self, image):
        """
        Process a single image frame for robotic perception
        """
        # Convert image to tensor
        input_tensor = self.preprocess(Image.fromarray(image))
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            # Extract visual features
            features = self.feature_extractor(input_batch)

            # Run object detection
            detections = self.object_detector(image)

            # Run semantic segmentation
            segmentation = self.segmentation_model(image)

            # Estimate depth
            depth_map = self.depth_estimator(image)

        return {
            'features': features,
            'detections': detections,
            'segmentation': segmentation,
            'depth': depth_map
        }

class ObjectDetectionModel(nn.Module):
    """
    Object detection model for robotic applications
    """
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        # This would typically be a pre-trained model like YOLO or Faster R-CNN
        # For this example, we'll create a simplified version
        self.backbone = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.detection_head = nn.Conv2d(64, 25, kernel_size=1)  # 25 = 4 bbox + 1 conf + 20 classes

    def forward(self, x):
        features = torch.relu(self.backbone(x))
        detections = self.detection_head(features)
        return detections

class SegmentationModel(nn.Module):
    """
    Semantic segmentation model for robotic applications
    """
    def __init__(self):
        super(SegmentationModel, self).__init__()
        # Simplified segmentation model
        self.encoder = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.decoder = nn.Conv2d(64, 21, kernel_size=1)  # 21 classes (20 objects + background)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        segmented = self.decoder(encoded)
        return segmented

class DepthEstimationModel(nn.Module):
    """
    Depth estimation model for robotic applications
    """
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        # Simplified depth estimation model
        self.encoder = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.decoder = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        depth = self.decoder(encoded)
        return depth
```

### Scene Understanding Techniques

Scene understanding in VLA systems involves comprehending not just individual objects but their relationships and the overall context:

- **Object Relationships**: Understanding spatial and functional relationships between objects
- **Scene Context**: Recognizing the broader scene category (kitchen, office, bedroom)
- **Affordance Detection**: Understanding what actions are possible with different objects
- **Spatial Layout**: Comprehending the 3D structure of the environment

### 3D Vision and Spatial Understanding

3D vision is crucial for robotics, enabling understanding of spatial relationships and supporting navigation and manipulation:

```python
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

class ThreeDVisionSystem:
    """
    3D vision and spatial understanding system for robotics
    """
    def __init__(self):
        self.point_cloud_processor = PointCloudProcessor()
        self.spatial_reasoner = SpatialReasoner()
        self.occupancy_mapper = OccupancyMapper()

    def process_3d_scene(self, rgb_image, depth_image, camera_intrinsics):
        """
        Process 3D scene from RGB-D input
        """
        # Convert to point cloud
        point_cloud = self.rgb_depth_to_pointcloud(rgb_image, depth_image, camera_intrinsics)

        # Process point cloud for 3D understanding
        scene_elements = self.point_cloud_processor.process(point_cloud)

        # Perform spatial reasoning
        spatial_relationships = self.spatial_reasoner.analyze(scene_elements)

        # Update occupancy map
        self.occupancy_mapper.update(point_cloud, spatial_relationships)

        return {
            'point_cloud': point_cloud,
            'scene_elements': scene_elements,
            'spatial_relationships': spatial_relationships,
            'occupancy_map': self.occupancy_mapper.get_map()
        }

    def rgb_depth_to_pointcloud(self, rgb_image, depth_image, intrinsics):
        """
        Convert RGB-D image to 3D point cloud
        """
        height, width = depth_image.shape
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]

        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to 3D coordinates
        x_3d = (x_coords - cx) * depth_image / fx
        y_3d = (y_coords - cy) * depth_image / fy
        z_3d = depth_image

        # Stack coordinates
        points = np.stack([x_3d, y_3d, z_3d], axis=-1).reshape(-1, 3)

        # Get colors
        colors = rgb_image.reshape(-1, 3) / 255.0

        # Remove invalid points (zero depth)
        valid_mask = points[:, 2] > 0
        points = points[valid_mask]
        colors = colors[valid_mask]

        return points, colors

class PointCloudProcessor:
    """
    Process point cloud data for 3D understanding
    """
    def __init__(self):
        self.segmentation_threshold = 0.05  # meters
        self.min_cluster_size = 100

    def process(self, point_cloud):
        """
        Process point cloud to extract 3D scene elements
        """
        points, colors = point_cloud

        # Downsample point cloud for efficiency
        downsampled = self.downsample(points, voxel_size=0.01)

        # Segment objects using region growing
        clusters = self.segment_objects(downsampled)

        # Extract features for each cluster
        scene_elements = []
        for i, cluster in enumerate(clusters):
            element = self.extract_element_features(cluster, i)
            scene_elements.append(element)

        return scene_elements

    def downsample(self, points, voxel_size=0.01):
        """
        Downsample point cloud using voxel grid
        """
        # Simple downsampling by voxelization
        voxel_coords = np.floor(points / voxel_size).astype(int)
        unique_coords, indices = np.unique(voxel_coords, axis=0, return_index=True)
        return points[indices]

    def segment_objects(self, points):
        """
        Segment objects using region growing approach
        """
        # This is a simplified approach
        # In practice, you'd use more sophisticated clustering like DBSCAN
        clusters = []
        visited = set()

        for i, point in enumerate(points):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            # Find nearby points
            distances = np.linalg.norm(points - point, axis=1)
            neighbors = np.where(distances < self.segmentation_threshold)[0]

            for neighbor in neighbors:
                if neighbor not in visited:
                    cluster.append(neighbor)
                    visited.add(neighbor)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(points[cluster])

        return clusters

    def extract_element_features(self, cluster, element_id):
        """
        Extract features for a scene element
        """
        centroid = np.mean(cluster, axis=0)
        bounding_box = self.compute_bounding_box(cluster)
        volume = self.compute_volume(cluster)
        shape_descriptor = self.compute_shape_descriptor(cluster)

        return {
            'id': element_id,
            'centroid': centroid,
            'bbox': bounding_box,
            'volume': volume,
            'shape': shape_descriptor,
            'points': cluster
        }

    def compute_bounding_box(self, points):
        """
        Compute oriented bounding box for a point cluster
        """
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        return {'min': min_pt, 'max': max_pt, 'center': (min_pt + max_pt) / 2}

    def compute_volume(self, points):
        """
        Compute approximate volume of a point cluster
        """
        bbox = self.compute_bounding_box(points)
        size = bbox['max'] - bbox['min']
        return np.prod(size)

    def compute_shape_descriptor(self, points):
        """
        Compute shape descriptor using PCA
        """
        # Center points
        centered = points - np.mean(points, axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)

        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalues
        sort_idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sort_idx]
        eigenvecs = eigenvecs[:, sort_idx]

        return {
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'elongation': eigenvals[0] / (eigenvals[1] + 1e-8),
            'planarity': (eigenvals[1] - eigenvals[2]) / (eigenvals[0] + 1e-8)
        }

class SpatialReasoner:
    """
    Perform spatial reasoning on scene elements
    """
    def __init__(self):
        self.support_threshold = 0.02  # 2cm threshold for support
        self.adjacency_threshold = 0.1  # 10cm threshold for adjacency

    def analyze(self, scene_elements):
        """
        Analyze spatial relationships between scene elements
        """
        relationships = []

        for i, elem1 in enumerate(scene_elements):
            for j, elem2 in enumerate(scene_elements):
                if i == j:
                    continue

                # Check for spatial relationships
                rels = self.compute_relationships(elem1, elem2)
                relationships.extend(rels)

        return relationships

    def compute_relationships(self, elem1, elem2):
        """
        Compute spatial relationships between two elements
        """
        rels = []

        # Distance-based relationships
        dist = np.linalg.norm(elem1['centroid'] - elem2['centroid'])

        # Support relationship (element2 supports element1)
        if self.check_support(elem1, elem2):
            rels.append({
                'type': 'support',
                'subject': elem1['id'],
                'object': elem2['id'],
                'confidence': 1.0
            })

        # Adjacency relationship
        if dist < self.adjacency_threshold:
            rels.append({
                'type': 'adjacent',
                'subject': elem1['id'],
                'object': elem2['id'],
                'distance': dist,
                'confidence': 1.0 - (dist / self.adjacency_threshold)
            })

        # Above/below relationship
        if abs(elem1['centroid'][2] - elem2['centroid'][2]) > 0.05:
            if elem1['centroid'][2] > elem2['centroid'][2]:
                rels.append({
                    'type': 'above',
                    'subject': elem1['id'],
                    'object': elem2['id'],
                    'confidence': 0.9
                })
            else:
                rels.append({
                    'type': 'below',
                    'subject': elem1['id'],
                    'object': elem2['id'],
                    'confidence': 0.9
                })

        return rels

    def check_support(self, top_elem, bottom_elem):
        """
        Check if bottom element supports top element
        """
        # Check if top element is above bottom element
        if top_elem['centroid'][2] <= bottom_elem['centroid'][2]:
            return False

        # Check if top element is close to bottom element vertically
        if top_elem['centroid'][2] - bottom_elem['centroid'][2] > self.support_threshold:
            return False

        # Check if top element's projection overlaps with bottom element's projection
        top_bbox = top_elem['bbox']
        bottom_bbox = bottom_elem['bbox']

        # 2D overlap check (x, y plane)
        x_overlap = max(0, min(top_bbox['max'][0], bottom_bbox['max'][0]) -
                           max(top_bbox['min'][0], bottom_bbox['min'][0]))
        y_overlap = max(0, min(top_bbox['max'][1], bottom_bbox['max'][1]) -
                           max(top_bbox['min'][1], bottom_bbox['min'][1]))

        return x_overlap > 0 and y_overlap > 0

class OccupancyMapper:
    """
    Maintain 3D occupancy map of the environment
    """
    def __init__(self, resolution=0.05):
        self.resolution = resolution
        self.occupancy_grid = {}  # Dictionary-based sparse grid
        self.origin = np.zeros(3)

    def update(self, point_cloud, spatial_relationships):
        """
        Update occupancy map with new point cloud data
        """
        points, colors = point_cloud

        # Convert points to grid coordinates
        grid_coords = np.floor((points - self.origin) / self.resolution).astype(int)

        # Update occupancy probabilities
        for coord in grid_coords:
            key = tuple(coord)
            # Update occupancy probability using probabilistic model
            current_prob = self.occupancy_grid.get(key, 0.5)
            new_prob = self.update_probability(current_prob, occupied=True)
            self.occupancy_grid[key] = new_prob

    def update_probability(self, current_prob, occupied=True, sensor_model_prob=0.7):
        """
        Update occupancy probability using Bayesian update
        """
        if occupied:
            # Sensor indicates occupied
            numerator = sensor_model_prob * current_prob
            denominator = (sensor_model_prob * current_prob +
                          (1 - sensor_model_prob) * (1 - current_prob))
        else:
            # Sensor indicates free
            numerator = (1 - sensor_model_prob) * current_prob
            denominator = ((1 - sensor_model_prob) * current_prob +
                          sensor_model_prob * (1 - current_prob))

        if denominator > 0:
            new_prob = numerator / denominator
        else:
            new_prob = current_prob

        return np.clip(new_prob, 0.001, 0.999)  # Avoid numerical issues

    def get_map(self):
        """
        Get current occupancy map
        """
        return self.occupancy_grid
```

## Multimodal Fusion Architectures

Multimodal fusion is the process of combining information from different modalities to create a unified representation.

### Fusion Strategies

There are several approaches to fusing information from vision and language:

**Early Fusion**: Combining raw or low-level features from different modalities early in the processing pipeline.

**Late Fusion**: Processing each modality separately and combining high-level representations late in the pipeline.

**Intermediate Fusion**: Combining information at multiple levels throughout the processing hierarchy.

### Transformer-Based Fusion Approaches

Transformer architectures have proven highly effective for multimodal fusion due to their attention mechanisms that can learn cross-modal relationships:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalTransformer(nn.Module):
    """
    Transformer-based multimodal fusion architecture
    """
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super(MultimodalTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        # Modality-specific encoders
        self.visual_encoder = VisualEncoder(d_model)
        self.text_encoder = TextEncoder(d_model)

        # Cross-modal attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model, nhead)
            for _ in range(num_layers)
        ])

        # Final fusion layer
        self.fusion_layer = nn.Linear(d_model * 2, d_model)

        # Output heads for different tasks
        self.vision_head = nn.Linear(d_model, 1000)  # Example: object classification
        self.language_head = nn.Linear(d_model, 30000)  # Example: word prediction
        self.action_head = nn.Linear(d_model, 128)  # Example: action parameters

    def forward(self, visual_features, text_features):
        """
        Forward pass through multimodal transformer
        """
        # Encode modalities
        vis_encoded = self.visual_encoder(visual_features)
        text_encoded = self.text_encoder(text_features)

        # Cross-modal attention processing
        for layer in self.cross_attention_layers:
            vis_encoded, text_encoded = layer(vis_encoded, text_encoded)

        # Final fusion
        fused_features = torch.cat([vis_encoded, text_encoded], dim=-1)
        fused_features = self.fusion_layer(fused_features)

        # Generate outputs
        vision_output = self.vision_head(fused_features)
        language_output = self.language_head(fused_features)
        action_output = self.action_head(fused_features)

        return {
            'fused_features': fused_features,
            'vision_output': vision_output,
            'language_output': language_output,
            'action_output': action_output
        }

class VisualEncoder(nn.Module):
    """
    Visual feature encoder
    """
    def __init__(self, d_model):
        super(VisualEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.projection = nn.Linear(256, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(49, d_model))  # 7x7 patches

    def forward(self, x):
        # Process visual features
        features = self.conv_layers(x)
        batch_size, channels, height, width = features.shape

        # Reshape to sequence
        features = features.view(batch_size, channels, -1).permute(0, 2, 1)  # (B, H*W, C)

        # Project to model dimension
        features = self.projection(features)

        # Add positional encoding
        features = features + self.positional_encoding[:features.size(1)]

        return features

class TextEncoder(nn.Module):
    """
    Text feature encoder
    """
    def __init__(self, d_model):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(30000, d_model)  # Vocabulary size of 30K
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8),
            num_layers=4
        )

    def forward(self, x):
        # x is token indices
        embedded = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        embedded = self.pos_encoding(embedded)
        encoded = self.transformer_layers(embedded)
        return encoded

class CrossModalAttentionLayer(nn.Module):
    """
    Cross-modal attention layer that allows vision and language to attend to each other
    """
    def __init__(self, d_model, nhead):
        super(CrossModalAttentionLayer, self).__init__()

        # Self-attention for each modality
        self.vis_self_attn = nn.MultiheadAttention(d_model, nhead)
        self.text_self_attn = nn.MultiheadAttention(d_model, nhead)

        # Cross-attention: vision attending to text
        self.vis_text_cross_attn = nn.MultiheadAttention(d_model, nhead)
        # Cross-attention: text attending to vision
        self.text_vis_cross_attn = nn.MultiheadAttention(d_model, nhead)

        # Feed-forward networks
        self.vis_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # Layer normalization
        self.vis_norm1 = nn.LayerNorm(d_model)
        self.vis_norm2 = nn.LayerNorm(d_model)
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

    def forward(self, vis_features, text_features):
        # Self-attention within each modality
        vis_self_out, _ = self.vis_self_attn(vis_features, vis_features, vis_features)
        text_self_out, _ = self.text_self_attn(text_features, text_features, text_features)

        # Add & Norm
        vis_features = self.vis_norm1(vis_features + vis_self_out)
        text_features = self.text_norm1(text_features + text_self_out)

        # Cross-attention: vision attends to text
        vis_text_out, _ = self.vis_text_cross_attn(vis_features, text_features, text_features)
        # Cross-attention: text attends to vision
        text_vis_out, _ = self.text_vis_cross_attn(text_features, vis_features, vis_features)

        # Add & Norm
        vis_features = self.vis_norm1(vis_features + vis_text_out)
        text_features = self.text_norm1(text_features + text_vis_out)

        # Feed-forward networks
        vis_ffn_out = self.vis_ffn(vis_features)
        text_ffn_out = self.text_ffn(text_features)

        # Add & Norm
        vis_features = self.vis_norm2(vis_features + vis_ffn_out)
        text_features = self.text_norm1(text_features + text_ffn_out)

        return vis_features, text_features

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Example usage
def create_multimodal_transformer():
    """
    Create a multimodal transformer for VLA applications
    """
    model = MultimodalTransformer(d_model=512, nhead=8, num_layers=6)
    return model
```

## Attention Mechanisms for VLA

Attention mechanisms are crucial for VLA systems, enabling selective focus on relevant information across modalities.

### Visual Attention in VLA Systems

Visual attention allows VLA systems to focus on relevant regions of the visual input based on language or action context:

```python
class VisualAttentionModule(nn.Module):
    """
    Visual attention module for VLA systems
    """
    def __init__(self, d_model=512):
        super(VisualAttentionModule, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.spatial_conv = nn.Conv2d(d_model, 1, kernel_size=1)

    def forward(self, visual_features, language_context=None):
        """
        Apply visual attention based on language context

        Args:
            visual_features: (batch_size, height*width, d_model)
            language_context: (batch_size, seq_len, d_model) or None
        """
        batch_size, num_patches, d_model = visual_features.shape

        # If language context is provided, use it as query for visual attention
        if language_context is not None:
            # Aggregate language context (e.g., mean pooling)
            lang_query = torch.mean(language_context, dim=1)  # (batch_size, d_model)

            # Project to visual feature space
            query = self.query_projection(lang_query).unsqueeze(1)  # (batch_size, 1, d_model)
            keys = self.key_projection(visual_features)  # (batch_size, num_patches, d_model)
            values = self.value_projection(visual_features)  # (batch_size, num_patches, d_model)

            # Compute attention scores
            attention_scores = torch.bmm(query, keys.transpose(1, 2)) / (d_model ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)

            # Apply attention to values
            attended_visual = torch.bmm(attention_weights, values)  # (batch_size, 1, d_model)

            # Expand back to patch dimension
            attended_visual = attended_visual.expand(-1, num_patches, -1)

        else:
            # Use spatial features for attention
            # Reshape to 2D feature map for spatial convolution
            height = width = int(num_patches ** 0.5)
            spatial_features = visual_features.view(batch_size, height, width, d_model).permute(0, 3, 1, 2)

            spatial_attention = torch.sigmoid(self.spatial_conv(spatial_features))
            spatial_attention = spatial_attention.permute(0, 2, 3, 1).view(batch_size, num_patches, 1)

            attended_visual = visual_features * spatial_attention

        return attended_visual, attention_weights if language_context is not None else spatial_attention

class LanguageGuidedAttention(nn.Module):
    """
    Language-guided visual attention
    """
    def __init__(self, d_model=512):
        super(LanguageGuidedAttention, self).__init__()

        self.language_encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.visual_attention = VisualAttentionModule(d_model)
        self.modality_fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, visual_features, language_tokens):
        """
        Apply language-guided attention to visual features
        """
        # Encode language
        lang_encoded, _ = self.language_encoder(language_tokens)

        # Apply language-guided visual attention
        attended_visual, attention_weights = self.visual_attention(
            visual_features, lang_encoded
        )

        # Fuse attended visual features with language context
        # For simplicity, we'll use the last language state
        final_lang_state = lang_encoded[:, -1, :]  # (batch_size, d_model)
        final_lang_state = final_lang_state.unsqueeze(1).expand(-1, attended_visual.size(1), -1)

        fused_features = torch.cat([attended_visual, final_lang_state], dim=-1)
        output_features = self.modality_fusion(fused_features)

        return output_features, attention_weights
```

### Multimodal Attention Visualization

Visualizing attention patterns helps understand how VLA systems process information:

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(visual_features, attention_weights, image, language_tokens):
    """
    Visualize attention weights overlaid on the original image
    """
    # Reshape visual features and attention weights
    batch_size, num_patches, d_model = visual_features.shape
    patch_size = int(np.sqrt(num_patches))

    # Reshape attention weights to spatial dimensions
    attention_map = attention_weights.view(batch_size, patch_size, patch_size)

    # Upsample attention map to image resolution
    attention_map = F.interpolate(
        attention_map.unsqueeze(1),
        size=image.shape[1:3],
        mode='bilinear',
        align_corners=False
    ).squeeze(1)

    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Attention map
    im1 = axes[1].imshow(attention_map[0], cmap='hot')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attention_map[0], cmap='hot', alpha=0.5)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
```

## Scene Understanding Systems

Scene understanding goes beyond object detection to comprehend the overall context and relationships within a scene.

### Holistic Scene Understanding

Holistic scene understanding involves:

- **Scene categorization**: Understanding the overall scene type (kitchen, office, etc.)
- **Object relationships**: Understanding how objects relate to each other spatially and functionally
- **Activity recognition**: Understanding what activities are happening or can happen
- **Contextual awareness**: Understanding the broader situation and implications

### Spatial and Temporal Reasoning

VLA systems must understand spatial relationships and how scenes evolve over time:

```python
class SceneUnderstandingSystem:
    """
    Scene understanding system for VLA applications
    """
    def __init__(self):
        self.scene_classifier = SceneClassifier()
        self.relationship_analyzer = RelationshipAnalyzer()
        self.activity_detector = ActivityDetector()
        self.temporal_reasoner = TemporalReasoner()

        # Scene context memory
        self.scene_memory = SceneMemory()

    def understand_scene(self, visual_features, language_input, robot_state):
        """
        Perform holistic scene understanding
        """
        # Classify overall scene
        scene_category = self.scene_classifier.classify(visual_features)

        # Analyze object relationships
        relationships = self.relationship_analyzer.analyze(visual_features)

        # Detect ongoing activities
        activities = self.activity_detector.detect(visual_features, robot_state)

        # Perform temporal reasoning
        temporal_context = self.temporal_reasoner.reason(
            visual_features,
            self.scene_memory.get_recent_states()
        )

        # Integrate all information
        scene_understanding = {
            'scene_category': scene_category,
            'relationships': relationships,
            'activities': activities,
            'temporal_context': temporal_context,
            'action_affordances': self.compute_action_affordances(relationships, robot_state)
        }

        # Update scene memory
        self.scene_memory.update(scene_understanding)

        return scene_understanding

    def compute_action_affordances(self, relationships, robot_state):
        """
        Compute what actions are affordanced by the current scene
        """
        affordances = []

        for relationship in relationships:
            if relationship['type'] == 'adjacent':
                obj1_id, obj2_id = relationship['subject'], relationship['object']
                distance = relationship['distance']

                # Check if objects are manipulable
                if self.is_manipulable(obj1_id) and distance < 0.5:  # Within reach
                    affordances.append({
                        'action': 'grasp',
                        'object': obj1_id,
                        'feasibility': 1.0 - min(distance / 0.5, 1.0),  # Closer = more feasible
                        'preconditions': ['free_space', 'no_obstacles']
                    })

                if self.is_surface(obj2_id) and self.is_small_object(obj1_id):
                    affordances.append({
                        'action': 'place_on',
                        'object': obj1_id,
                        'surface': obj2_id,
                        'feasibility': 0.8,
                        'preconditions': ['surface_free', 'stable_placement']
                    })

        return affordances

    def is_manipulable(self, obj_id):
        """
        Check if an object is manipulable by the robot
        """
        # This would typically involve checking object properties
        # like size, weight, shape, etc.
        return True  # Simplified for example

    def is_surface(self, obj_id):
        """
        Check if an object is a surface (table, counter, etc.)
        """
        return True  # Simplified for example

    def is_small_object(self, obj_id):
        """
        Check if an object is small enough to be grasped
        """
        return True  # Simplified for example

class SceneClassifier(nn.Module):
    """
    Scene category classifier
    """
    def __init__(self, num_categories=10):
        super(SceneClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_categories)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class RelationshipAnalyzer:
    """
    Analyze relationships between objects in a scene
    """
    def __init__(self):
        self.spatial_analyzer = SpatialRelationshipAnalyzer()
        self.functional_analyzer = FunctionalRelationshipAnalyzer()

    def analyze(self, visual_features):
        """
        Analyze relationships in the scene
        """
        spatial_relationships = self.spatial_analyzer.analyze(visual_features)
        functional_relationships = self.functional_analyzer.analyze(visual_features)

        return spatial_relationships + functional_relationships

class SpatialRelationshipAnalyzer:
    """
    Analyze spatial relationships between objects
    """
    def analyze(self, visual_features):
        """
        Analyze spatial relationships (above, below, left, right, adjacent, etc.)
        """
        relationships = []

        # This would typically involve object detection and spatial reasoning
        # For simplicity, we'll create example relationships
        relationships.append({
            'type': 'above',
            'subject': 'mug',
            'object': 'table',
            'confidence': 0.95
        })
        relationships.append({
            'type': 'left_of',
            'subject': 'phone',
            'object': 'computer',
            'confidence': 0.87
        })

        return relationships

class FunctionalRelationshipAnalyzer:
    """
    Analyze functional relationships between objects
    """
    def analyze(self, visual_features):
        """
        Analyze functional relationships (used_with, contained_in, etc.)
        """
        relationships = []

        # This would typically involve understanding object functions
        relationships.append({
            'type': 'used_with',
            'subject': 'coffee_mug',
            'object': 'coffee_machine',
            'confidence': 0.92
        })
        relationships.append({
            'type': 'contains',
            'subject': 'drawer',
            'object': 'utensils',
            'confidence': 0.89
        })

        return relationships

class ActivityDetector:
    """
    Detect activities happening in the scene
    """
    def __init__(self):
        self.activity_templates = self.load_activity_templates()

    def load_activity_templates(self):
        """
        Load templates for common activities
        """
        return {
            'cooking': ['knife', 'cutting_board', 'ingredients'],
            'working': ['computer', 'keyboard', 'documents'],
            'eating': ['plate', 'fork', 'food']
        }

    def detect(self, visual_features, robot_state):
        """
        Detect activities based on scene content
        """
        detected_activities = []

        # This would typically involve activity recognition models
        # For simplicity, we'll match against templates
        for activity, objects in self.activity_templates.items():
            # Check if scene contains objects relevant to this activity
            scene_objects = self.extract_objects(visual_features)
            matching_objects = [obj for obj in objects if obj in scene_objects]

            if len(matching_objects) >= len(objects) * 0.5:  # At least 50% match
                detected_activities.append({
                    'activity': activity,
                    'confidence': len(matching_objects) / len(objects),
                    'relevant_objects': matching_objects
                })

        return detected_activities

    def extract_objects(self, visual_features):
        """
        Extract object information from visual features
        """
        # This would typically involve object detection
        return ['mug', 'table', 'phone', 'computer']  # Example objects

class TemporalReasoner:
    """
    Reason about temporal aspects of the scene
    """
    def __init__(self):
        self.temporal_patterns = {}

    def reason(self, current_features, past_states):
        """
        Perform temporal reasoning based on current and past states
        """
        if not past_states:
            return {'change_detection': [], 'predicted_actions': []}

        # Detect changes from previous states
        changes = self.detect_changes(current_features, past_states[-1])

        # Predict likely future actions based on observed patterns
        predicted_actions = self.predict_future_actions(changes, past_states)

        return {
            'changes': changes,
            'predicted_actions': predicted_actions,
            'temporal_consistency': self.assess_temporal_consistency(past_states)
        }

    def detect_changes(self, current_features, previous_features):
        """
        Detect changes between current and previous states
        """
        changes = []
        # Implementation would compare current and previous states
        return changes

    def predict_future_actions(self, changes, past_states):
        """
        Predict likely future actions based on observed changes
        """
        predictions = []
        # Implementation would use temporal patterns to predict actions
        return predictions

    def assess_temporal_consistency(self, past_states):
        """
        Assess consistency of scene over time
        """
        return 1.0  # Perfect consistency for now

class SceneMemory:
    """
    Memory system for tracking scene states over time
    """
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.states = []

    def update(self, scene_state):
        """
        Update scene memory with new state
        """
        self.states.append(scene_state)
        if len(self.states) > self.capacity:
            self.states.pop(0)  # Remove oldest state

    def get_recent_states(self, n=5):
        """
        Get n most recent scene states
        """
        return self.states[-n:] if len(self.states) >= n else self.states

    def find_similar_scenes(self, query_state, threshold=0.8):
        """
        Find similar scenes in memory
        """
        similar_scenes = []
        for state in self.states:
            similarity = self.compute_scene_similarity(query_state, state)
            if similarity > threshold:
                similar_scenes.append((state, similarity))

        return sorted(similar_scenes, key=lambda x: x[1], reverse=True)

    def compute_scene_similarity(self, state1, state2):
        """
        Compute similarity between two scene states
        """
        # Simplified similarity computation
        return 0.5  # Placeholder
```

## Object Detection with Language Grounding

Language-grounded object detection connects linguistic descriptions to visual objects, enabling more natural human-robot interaction.

### Object Detection with Language Context

Language grounding allows the system to detect objects based on linguistic descriptions rather than just visual patterns:

```python
class LanguageGroundedDetector(nn.Module):
    """
    Language-grounded object detection system
    """
    def __init__(self, num_classes=91, d_model=512):
        super(LanguageGroundedDetector, self).__init__()

        # Visual backbone
        self.backbone = self._build_backbone()

        # Language encoder
        self.lang_encoder = TextEncoder(d_model)

        # Detection head
        self.detection_head = DetectionHead(d_model, num_classes)

        # Language grounding module
        self.grounding_module = LanguageGroundingModule(d_model)

        # Feature fusion
        self.fusion_module = nn.Linear(d_model * 2, d_model)

    def _build_backbone(self):
        """
        Build visual backbone for feature extraction
        """
        import torchvision.models as models
        backbone = models.resnet50(pretrained=True)
        # Remove classification layer
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        return backbone

    def forward(self, images, language_queries, return_features=False):
        """
        Forward pass for language-grounded detection
        """
        batch_size, _, H, W = images.shape

        # Extract visual features
        visual_features = self.backbone(images)  # (B, C, H', W')

        # Encode language queries
        lang_features = self.lang_encoder(language_queries)  # (B, seq_len, d_model)

        # Apply language grounding to visual features
        grounded_features = self.grounding_module(visual_features, lang_features)

        # Detect objects
        detections = self.detection_head(grounded_features)

        if return_features:
            return detections, grounded_features
        else:
            return detections

class LanguageGroundingModule(nn.Module):
    """
    Module for grounding language in visual features
    """
    def __init__(self, d_model):
        super(LanguageGroundingModule, self).__init__()

        self.attention = nn.MultiheadAttention(d_model, num_heads=8)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        # Positional encoding for visual features
        self.pos_encoding = PositionalEncoding(d_model)

    def forward(self, visual_features, language_features):
        """
        Ground language features in visual features
        """
        batch_size, C, H, W = visual_features.shape

        # Reshape visual features to (H*W, B, C) for attention
        visual_flat = visual_features.view(batch_size, C, H*W).permute(2, 0, 1)

        # Add positional encoding to visual features
        visual_flat = visual_flat + self.pos_encoding(visual_flat)

        # Language features: (seq_len, B, d_model)
        lang_seq_len, _, d_model = language_features.shape
        language_features = language_features.permute(1, 0, 2)  # (B, seq_len, d_model)

        # Cross-attention: visual features attend to language
        attended_visual, attention_weights = self.attention(
            visual_flat,  # query
            language_features.transpose(0, 1),  # key
            language_features.transpose(0, 1)   # value
        )

        # Reshape back to spatial format
        attended_visual = attended_visual.permute(1, 2, 0)  # (B, C, H*W)
        attended_visual = attended_visual.view(batch_size, C, H, W)

        # Residual connection and normalization
        output = self.norm(visual_features + self.dropout(attended_visual))

        return output

class DetectionHead(nn.Module):
    """
    Detection head for language-grounded detection
    """
    def __init__(self, d_model, num_classes):
        super(DetectionHead, self).__init__()

        # Separate heads for different detection tasks
        self.classifier = nn.Conv2d(d_model, num_classes, kernel_size=1)
        self.bbox_reg = nn.Conv2d(d_model, 4, kernel_size=1)  # dx, dy, dw, dh
        self.objectness = nn.Conv2d(d_model, 1, kernel_size=1)

    def forward(self, features):
        """
        Generate detection outputs
        """
        class_logits = self.classifier(features)
        bbox_deltas = self.bbox_reg(features)
        objectness = self.objectness(features)

        return {
            'class_logits': class_logits,
            'bbox_deltas': bbox_deltas,
            'objectness': objectness
        }

class ReferringExpressionComprehension(nn.Module):
    """
    System for understanding referring expressions (e.g., "the red mug on the table")
    """
    def __init__(self, vocab_size=30000, d_model=512):
        super(ReferringExpressionComprehension, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Text encoder
        self.text_encoder = nn.Embedding(vocab_size, d_model)
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8),
            num_layers=4
        )

        # Visual encoder
        self.visual_encoder = VisualEncoder(d_model)

        # Cross-modal matching
        self.matching_module = nn.Linear(d_model * 2, 1)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

    def forward(self, visual_features, text_tokens, candidate_objects):
        """
        Comprehend referring expressions and localize target objects

        Args:
            visual_features: (B, C, H, W)
            text_tokens: (B, seq_len) - tokenized referring expression
            candidate_objects: List of candidate object proposals
        """
        batch_size = visual_features.size(0)

        # Encode text
        text_embeddings = self.text_encoder(text_tokens)  # (B, seq_len, d_model)
        text_embeddings = text_embeddings + self.pos_encoding(text_embeddings)
        text_features = self.text_transformer(text_embeddings)  # (B, seq_len, d_model)

        # Pool text features to get sentence-level representation
        sentence_features = torch.mean(text_features, dim=1)  # (B, d_model)

        # Encode visual features
        visual_features = self.visual_encoder(visual_features)  # (B, num_patches, d_model)

        # For each candidate object, compute match score with text
        scores = []
        for obj_proposal in candidate_objects:
            # Extract visual features for this object
            obj_features = self.extract_object_features(visual_features, obj_proposal)

            # Compute cross-modal matching score
            concat_features = torch.cat([sentence_features, obj_features], dim=-1)
            score = self.matching_module(concat_features)
            scores.append(score)

        # Stack scores
        scores = torch.stack(scores, dim=1)  # (B, num_candidates)

        # Apply softmax to get probabilities
        probabilities = F.softmax(scores, dim=1)

        return probabilities

    def extract_object_features(self, visual_features, proposal):
        """
        Extract visual features for a specific object proposal
        """
        # This would typically involve ROI pooling or similar
        # For simplicity, we'll return mean pooled features
        return torch.mean(visual_features, dim=1)  # (B, d_model)

class OpenVocabularyDetector(nn.Module):
    """
    Open vocabulary object detection using language models
    """
    def __init__(self, clip_model, d_model=512):
        super(OpenVocabularyDetector, self).__init__()

        self.clip_model = clip_model  # Pre-trained CLIP model
        self.detection_backbone = self._build_detection_backbone()
        self.classifier = nn.Linear(d_model, d_model)  # Will be used with text features

    def _build_detection_backbone(self):
        """
        Build detection backbone (similar to DETR or FCOS)
        """
        # This is a simplified version
        # In practice, you'd use a more sophisticated architecture
        return nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, images, text_descriptions):
        """
        Detect objects with open vocabulary using text descriptions

        Args:
            images: Input images (B, 3, H, W)
            text_descriptions: List of text descriptions for each image
        """
        # Extract visual features
        visual_features = self.detection_backbone(images)

        # Encode text descriptions using CLIP text encoder
        text_features = self.encode_texts(text_descriptions)

        # Compute similarity between visual features and text descriptions
        batch_size, d_model, H, W = visual_features.shape
        visual_flat = visual_features.view(batch_size, d_model, -1).permute(0, 2, 1)  # (B, HW, d_model)

        # Compute similarity scores
        similarity_scores = torch.bmm(visual_flat, text_features.transpose(1, 2))  # (B, HW, num_texts)

        # Reshape back to spatial format
        similarity_maps = similarity_scores.view(batch_size, -1, H, W)  # (B, num_texts, H, W)

        return similarity_maps

    def encode_texts(self, text_descriptions):
        """
        Encode text descriptions using CLIP text encoder
        """
        # This is a placeholder - in practice, you'd use the actual CLIP text encoder
        # For now, we'll just return dummy features
        max_descriptions = max(len(descs) for descs in text_descriptions)
        batch_size = len(text_descriptions)

        # Dummy text features
        text_features = torch.randn(batch_size, max_descriptions, 512)
        return text_features
```

## Spatial Reasoning Systems

Spatial reasoning enables VLA systems to understand and manipulate spatial relationships in the environment.

### Spatial Reasoning in VLA Contexts

Spatial reasoning in VLA systems encompasses:

- **Topological reasoning**: Understanding spatial relationships like adjacency, containment, and connectivity
- **Metric reasoning**: Understanding precise distances and sizes
- **Geometric reasoning**: Understanding shapes, orientations, and transformations
- **Navigational reasoning**: Understanding paths and routes through space

### Geometric Reasoning Capabilities

Geometric reasoning allows the system to understand shapes, sizes, and spatial configurations:

```python
import math

class SpatialReasoningSystem:
    """
    Spatial reasoning system for VLA applications
    """
    def __init__(self):
        self.topological_reasoner = TopologicalReasoner()
        self.metric_reasoner = MetricReasoner()
        self.geometric_reasoner = GeometricReasoner()
        self.navigation_reasoner = NavigationReasoner()

    def reason_about_space(self, scene_elements, spatial_constraints=None):
        """
        Perform comprehensive spatial reasoning about the scene
        """
        topological_relations = self.topological_reasoner.analyze(scene_elements)
        metric_properties = self.metric_reasoner.analyze(scene_elements)
        geometric_properties = self.geometric_reasoner.analyze(scene_elements)
        navigable_paths = self.navigation_reasoner.find_paths(scene_elements, spatial_constraints)

        spatial_knowledge = {
            'topological': topological_relations,
            'metric': metric_properties,
            'geometric': geometric_properties,
            'navigational': navigable_paths
        }

        return spatial_knowledge

class TopologicalReasoner:
    """
    Reason about topological relationships (connectedness, containment, adjacency)
    """
    def analyze(self, scene_elements):
        """
        Analyze topological relationships between scene elements
        """
        relations = []

        for i, elem1 in enumerate(scene_elements):
            for j, elem2 in enumerate(scene_elements):
                if i == j:
                    continue

                rel = self.compute_topological_relation(elem1, elem2)
                if rel:
                    relations.append(rel)

        return relations

    def compute_topological_relation(self, elem1, elem2):
        """
        Compute topological relation between two elements
        """
        # Check for different topological relationships
        if self.is_adjacent(elem1, elem2):
            return {
                'type': 'adjacent',
                'subject': elem1['id'],
                'object': elem2['id'],
                'confidence': 0.9
            }
        elif self.is_connected(elem1, elem2):
            return {
                'type': 'connected',
                'subject': elem1['id'],
                'object': elem2['id'],
                'confidence': 0.85
            }
        elif self.is_inside(elem1, elem2):
            return {
                'type': 'inside',
                'subject': elem1['id'],
                'object': elem2['id'],
                'confidence': 0.95
            }
        elif self.is_contains(elem1, elem2):
            return {
                'type': 'contains',
                'subject': elem1['id'],
                'object': elem2['id'],
                'confidence': 0.95
            }

        return None

    def is_adjacent(self, elem1, elem2, threshold=0.1):
        """
        Check if two elements are adjacent
        """
        center1 = elem1['centroid']
        center2 = elem2['centroid']
        distance = np.linalg.norm(center1 - center2)

        # Consider adjacency based on distance and object sizes
        size1 = elem1.get('size', np.array([0.1, 0.1, 0.1]))
        size2 = elem2.get('size', np.array([0.1, 0.1, 0.1]))

        min_distance = (np.max(size1) + np.max(size2)) / 2

        return distance < (min_distance + threshold)

    def is_connected(self, elem1, elem2):
        """
        Check if two elements are connected (e.g., door connected to room)
        """
        # This would typically involve semantic knowledge
        # For now, we'll use spatial proximity with semantic rules
        return False  # Placeholder

    def is_inside(self, elem1, elem2):
        """
        Check if elem1 is inside elem2
        """
        # Check if elem1's centroid is within elem2's bounding box
        elem2_bbox = elem2['bbox']
        elem1_center = elem1['centroid']

        return (elem2_bbox['min'][0] <= elem1_center[0] <= elem2_bbox['max'][0] and
                elem2_bbox['min'][1] <= elem1_center[1] <= elem2_bbox['max'][1] and
                elem2_bbox['min'][2] <= elem1_center[2] <= elem2_bbox['max'][2])

    def is_contains(self, elem1, elem2):
        """
        Check if elem1 contains elem2
        """
        return self.is_inside(elem2, elem1)

class MetricReasoner:
    """
    Reason about metric properties (distances, sizes, volumes)
    """
    def analyze(self, scene_elements):
        """
        Analyze metric properties of scene elements
        """
        metric_analysis = []

        for elem in scene_elements:
            properties = {
                'id': elem['id'],
                'position': elem['centroid'].tolist(),
                'size': self.compute_size(elem),
                'volume': elem.get('volume', 0.0),
                'distances_to_others': self.compute_distances(elem, scene_elements)
            }
            metric_analysis.append(properties)

        return metric_analysis

    def compute_size(self, element):
        """
        Compute size of an element
        """
        bbox = element['bbox']
        size = bbox['max'] - bbox['min']
        return size.tolist()

    def compute_distances(self, element, all_elements):
        """
        Compute distances from element to all other elements
        """
        distances = {}
        for other_elem in all_elements:
            if other_elem['id'] != element['id']:
                dist = np.linalg.norm(element['centroid'] - other_elem['centroid'])
                distances[other_elem['id']] = dist
        return distances

class GeometricReasoner:
    """
    Reason about geometric properties (shapes, orientations, symmetries)
    """
    def analyze(self, scene_elements):
        """
        Analyze geometric properties of scene elements
        """
        geometric_analysis = []

        for elem in scene_elements:
            properties = {
                'id': elem['id'],
                'shape': self.classify_shape(elem),
                'orientation': self.estimate_orientation(elem),
                'symmetry': self.assess_symmetry(elem),
                'stability': self.assess_stability(elem)
            }
            geometric_analysis.append(properties)

        return geometric_analysis

    def classify_shape(self, element):
        """
        Classify the shape of an element based on its features
        """
        shape_features = element.get('shape', {})
        eigenvals = shape_features.get('eigenvalues', np.array([1, 1, 1]))

        # Classify based on eigenvalue ratios
        if eigenvals[0] / (eigenvals[1] + 1e-8) > 10:
            return 'line'  # Elongated
        elif eigenvals[1] / (eigenvals[2] + 1e-8) < 2 and eigenvals[0] / (eigenvals[2] + 1e-8) < 5:
            return 'sphere'  # Approximately spherical
        elif eigenvals[0] / (eigenvals[2] + 1e-8) > 5 and eigenvals[1] / (eigenvals[2] + 1e-8) < 3:
            return 'plane'  # Flat
        else:
            return 'irregular'

    def estimate_orientation(self, element):
        """
        Estimate the orientation of an element
        """
        shape_features = element.get('shape', {})
        eigenvectors = shape_features.get('eigenvectors', np.eye(3))

        # The principal axis is the eigenvector corresponding to the largest eigenvalue
        principal_axis = eigenvectors[:, 0]  # First column is principal axis

        # Convert to Euler angles for interpretation
        euler_angles = self.rotation_matrix_to_euler(eigenvectors)

        return {
            'principal_axis': principal_axis.tolist(),
            'euler_angles': euler_angles
        }

    def rotation_matrix_to_euler(self, R):
        """
        Convert rotation matrix to Euler angles (simplified)
        """
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return [x, y, z]

    def assess_symmetry(self, element):
        """
        Assess the symmetry of an element
        """
        # Simplified symmetry assessment based on eigenvalue ratios
        shape_features = element.get('shape', {})
        eigenvals = shape_features.get('eigenvalues', np.array([1, 1, 1]))

        # More symmetric if eigenvalues are closer to each other
        variance = np.var(eigenvals)
        symmetry_score = 1.0 / (1.0 + variance)  # Higher variance = lower symmetry

        return symmetry_score

    def assess_stability(self, element):
        """
        Assess the stability of an element based on its geometry
        """
        # Simplified stability assessment
        # Consider base area, center of mass height, etc.
        shape_features = element.get('shape', {})
        eigenvals = shape_features.get('eigenvalues', np.array([1, 1, 1]))

        # Stability is inversely related to the ratio of largest to smallest eigenvalues
        stability = eigenvals[2] / (eigenvals[0] + 1e-8)  # Z-axis (height) to X-axis (width) ratio

        # Lower values indicate more stability (wider base relative to height)
        return min(stability, 1.0)

class NavigationReasoner:
    """
    Reason about navigable paths and routes
    """
    def __init__(self):
        self.path_finder = PathFinder()

    def find_paths(self, scene_elements, constraints=None):
        """
        Find navigable paths through the scene
        """
        # Build navigation graph based on scene elements
        nav_graph = self.build_navigation_graph(scene_elements)

        # Find paths considering constraints
        paths = self.path_finder.find_paths(nav_graph, constraints)

        return paths

    def build_navigation_graph(self, scene_elements):
        """
        Build a navigation graph from scene elements
        """
        graph = {
            'nodes': [],
            'edges': [],
            'obstacles': []
        }

        # Add navigable locations
        for elem in scene_elements:
            if self.is_navigable_area(elem):
                graph['nodes'].append({
                    'id': elem['id'],
                    'position': elem['centroid'].tolist(),
                    'traversable': True
                })

        # Add obstacles
        for elem in scene_elements:
            if self.is_obstacle(elem):
                graph['obstacles'].append({
                    'id': elem['id'],
                    'bbox': elem['bbox'],
                    'traversable': False
                })

        # Add edges between nearby navigable areas
        for i, node1 in enumerate(graph['nodes']):
            for j, node2 in enumerate(graph['nodes']):
                if i != j:
                    dist = np.linalg.norm(
                        np.array(node1['position']) - np.array(node2['position'])
                    )
                    if dist < 2.0:  # Within 2 meters
                        graph['edges'].append({
                            'from': node1['id'],
                            'to': node2['id'],
                            'cost': dist
                        })

        return graph

    def is_navigable_area(self, element):
        """
        Check if an element represents a navigable area
        """
        # Typically floor, paths, open spaces
        shape = element.get('shape_class', 'unknown')
        return shape in ['floor', 'path', 'open_space']

    def is_obstacle(self, element):
        """
        Check if an element is an obstacle
        """
        # Typically furniture, walls, barriers
        shape = element.get('shape_class', 'unknown')
        return shape in ['furniture', 'wall', 'barrier']

class PathFinder:
    """
    Path finding algorithm for navigation
    """
    def find_paths(self, graph, constraints=None):
        """
        Find paths in the navigation graph
        """
        # For simplicity, we'll implement a basic A* algorithm
        # In practice, you'd use more sophisticated path planning
        if not graph['nodes']:
            return []

        # Find a path from the first node to the last node
        start_node = graph['nodes'][0]
        end_node = graph['nodes'][-1] if len(graph['nodes']) > 1 else start_node

        # Simple straight-line path for demonstration
        path = [start_node['position']]
        if start_node['id'] != end_node['id']:
            path.append(end_node['position'])

        return [{
            'path': path,
            'cost': self.estimate_path_cost(path),
            'feasible': True
        }]

    def estimate_path_cost(self, path):
        """
        Estimate the cost of a path
        """
        if len(path) < 2:
            return 0.0

        total_cost = 0.0
        for i in range(1, len(path)):
            dist = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
            total_cost += dist

        return total_cost
```

## Conclusion

Multimodal perception systems form the foundation of Vision-Language-Action (VLA) systems, enabling robots to understand their environment through multiple sensory channels and connect that understanding to linguistic descriptions and physical actions. The integration of visual perception, multimodal fusion, attention mechanisms, scene understanding, and spatial reasoning creates a comprehensive system capable of rich environmental interpretation.

The key components work together synergistically:

- Visual perception provides the raw sensory input
- Multimodal fusion combines information from different modalities
- Attention mechanisms focus processing on relevant information
- Scene understanding provides contextual awareness
- Object detection with language grounding enables natural interaction
- Spatial reasoning allows for navigation and manipulation

These systems continue to evolve rapidly, with ongoing research addressing challenges in real-time processing, robustness to environmental variations, and the integration of additional sensory modalities. The success of VLA systems depends critically on the quality of these perception components, making them a crucial area of focus for advanced robotics applications.