# import asyncio
# import websockets
# import json
# import time
# from datetime import datetime
# import os
#
# async def test_ai_websocket():
#     url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
#     headers = {
#         "OpenAI-Beta": "realtime=v1"
#     }
#     request_interval = 60  # 1분마다 새로운 분석 요청
#     last_request_time = 0
#
#     def format_time():
#         return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#     print(f"\n[{format_time()}] AI WebSocket 테스트 시작...")
#
#     try:
#         async with websockets.connect(url, extra_headers=headers) as websocket:
#             print(f"[{format_time()}] WebSocket 연결 성공")
#
#             # 초기 설정 메시지 전송
#             init_message = {
#                 "type": "response.create",
#                 "response": {
#                     "type": "text",
#                     "modalities": ["text"],
#                     "instructions": "한국어로 대화해주세요. 각 메시지에 자세히 답변해주세요."
#                 }
#             }
#             await websocket.send(json.dumps(init_message))
#             print(f"[{format_time()}] 초기 설정 메시지 전송됨")
#
#             # 응답 대기
#             await asyncio.sleep(2)
#
#             # 대화 메시지 전송
#             conversation_message = {
#                 "type": "conversation.item.create",
#                 "item": {
#                     "type": "message",
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "input_text",
#                             "text": "오늘 날씨가 어떤지 자세히 설명해주세요."
#                         }
#                     ]
#                 }
#             }
#
#             print(f"\n[{format_time()}] 대화 메시지 전송: {conversation_message}")
#             await websocket.send(json.dumps(conversation_message))
#
#             # 응답 처리
#             while True:
#                 try:
#                     response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
#                     response_data = json.loads(response)
#
#                     print(f"\n[{format_time()}] 새로운 응답:")
#                     print("-" * 50)
#
#                     # 실제 텍스트 응답 처리
#                     if response_data.get("type") == "response.output_item.content.added":
#                         delta = response_data.get("delta", {})
#                         if "text" in delta:
#                             print(f"AI 응답: {delta['text']}")
#                     else:
#                         print(json.dumps(response_data, indent=2, ensure_ascii=False))
#
#                     print("-" * 50)
#
#                     # 응답이 완료되었는지 확인
#                     if response_data.get("type") == "response.done":
#                         if response_data.get("response", {}).get("status") == "completed":
#                             print(f"\n[{format_time()}] 응답 완료")
#                             break
#                         elif response_data.get("response", {}).get("status") == "failed":
#                             error_details = response_data.get("response", {}).get("status_details", {}).get("error", {})
#                             print(f"\n[{format_time()}] 오류 발생: {error_details.get('message', '알 수 없는 오류')}")
#                             break
#
#                 except asyncio.TimeoutError:
#                     print(f"\n[{format_time()}] 응답 대기 시간 초과")
#                     break
#                 except Exception as e:
#                     print(f"\n[{format_time()}] 예외 발생: {str(e)}")
#                     break
#
#                 await asyncio.sleep(0.1)
#
#     except Exception as e:
#         print(f"\n[{format_time()}] 연결 오류: {str(e)}")
#     finally:
#         print(f"\n[{format_time()}] 프로그램 종료됨")
#
# if __name__ == "__main__":
#     # OPENAI_API_KEY 환경 변수 확인
#     # if not os.environ.get('OPENAI_API_KEY'):
#     #     print("Error: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
#     #     print("다음 명령어로 API 키를 설정해주세요:")
#     #     print("export OPENAI_API_KEY='your_api_key_here'")
#     #     exit(1)
#
#     # asyncio 이벤트 루프 실행
#     asyncio.run(test_ai_websocket())