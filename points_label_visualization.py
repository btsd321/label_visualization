# 多边形框可视化脚本
import os
import json
import cv2
import numpy as np

def visualize_labels(input_dir, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	for file in os.listdir(input_dir):
		if file.endswith('.json'):
			json_path = os.path.join(input_dir, file)
			img_name = file.replace('.json', '.jpg')
			img_path = os.path.join(input_dir, img_name)
			if not os.path.exists(img_path):
				print(f"Image not found for {file}")
				continue
			# 读取图片
			img = cv2.imread(img_path)
			# 读取json
			with open(json_path, 'r', encoding='utf-8') as f:
				data = json.load(f)
			# 兼容单个对象和对象列表
			if isinstance(data, dict) and 'points' in data:
				objects = [data]
			elif isinstance(data, list):
				objects = data
			else:
				print(f"Unknown JSON format in {file}")
				continue
			for obj in objects:
				points = obj.get('points', [])
				if not points:
					continue
				pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
				# 绘制多边形边界
				cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
				# 填充多边形（可选，半透明）
				overlay = img.copy()
				cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
				alpha = 0.3
				img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
				# 绘制标签文本
				label = None
				if 'type' in obj and isinstance(obj['type'], dict):
					label = list(obj['type'].values())[0]
				elif 'labelType' in obj:
					label = obj['labelType']
				if label:
					center = np.mean(pts, axis=0).astype(int)[0]
					cv2.putText(img, str(label), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
			# 保存结果
			out_path = os.path.join(output_dir, img_name)
			cv2.imwrite(out_path, img)
			print(f"Saved visualization: {out_path}")

if __name__ == "__main__":
	visualize_labels('input', 'output')