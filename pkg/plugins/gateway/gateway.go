/*
Copyright 2024 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	openai "github.com/sashabaranov/go-openai"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	"github.com/aibrix/aibrix/pkg/cache"
	routing "github.com/aibrix/aibrix/pkg/plugins/gateway/algorithms"
	ratelimiter "github.com/aibrix/aibrix/pkg/plugins/gateway/ratelimiter"
	"github.com/aibrix/aibrix/pkg/utils"
	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
)

var (
	routingStrategies = []string{"random", "least-request", "throughput"}
)

const (
	DefaultRPM           = 100
	DefaultTPMMultiplier = 1000
	InternalServerError  = envoyTypePb.StatusCode_InternalServerError
)

type Server struct {
	mu          sync.RWMutex
	routers     map[string]routing.Router
	redisClient *redis.Client
	ratelimiter ratelimiter.RateLimiter
	cache       *cache.Cache
}

func NewServer(redisClient *redis.Client) *Server {
	cache, err := cache.GetCache()
	if err != nil {
		panic(err)
	}
	r := ratelimiter.NewRedisAccountRateLimiter("aibrix", redisClient, 1*time.Minute)
	routers := map[string]routing.Router{
		"random":        routing.NewRandomRouter(),
		"least-request": routing.NewLeastRequestRouter(),
		"throughput":    routing.NewThroughputRouter(),
	}

	return &Server{
		routers:     routers,
		redisClient: redisClient,
		ratelimiter: r,
		cache:       cache,
	}
}

type HealthServer struct{}

func (s *HealthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *HealthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "watch is not implemented")
}

func (s *Server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	var user utils.User
	var routingStrategy, targetPodIP string
	ctx := srv.Context()
	requestID := uuid.New().String()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, err := srv.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", err)
		}

		resp := &extProcPb.ProcessingResponse{}
		switch v := req.Request.(type) {

		case *extProcPb.ProcessingRequest_RequestHeaders:
			resp, user, routingStrategy = s.HandleRequestHeaders(ctx, requestID, req)

		case *extProcPb.ProcessingRequest_RequestBody:
			resp, targetPodIP = s.HandleRequestBody(ctx, requestID, req, user, routingStrategy)

		case *extProcPb.ProcessingRequest_ResponseBody:
			resp = s.HandleResponseBody(ctx, requestID, req, user, targetPodIP)

		default:
			klog.Infof("Unknown Request type %+v\n", v)
		}

		if err := srv.Send(resp); err != nil {
			klog.Infof("send error %v", err)
		}
	}
}

func (s *Server) HandleRequestHeaders(ctx context.Context, requestID string, req *extProcPb.ProcessingRequest) (*extProcPb.ProcessingResponse, utils.User, string) {
	klog.Info("\n\n")
	klog.Info("-- In RequestHeaders processing ...")
	var username string
	var user utils.User
	var err error
	var errRes *extProcPb.ProcessingResponse

	h := req.Request.(*extProcPb.ProcessingRequest_RequestHeaders)
	for _, n := range h.RequestHeaders.Headers.Headers {
		if strings.ToLower(n.Key) == "user" {
			username = string(n.RawValue)
		}
	}

	routingStrategy, routingStrategyEnabled := getRoutingStrategy(h.RequestHeaders.Headers.Headers)
	if routingStrategyEnabled && !validateRoutingStrategy(routingStrategy) {
		return generateErrorResponse(
			envoyTypePb.StatusCode_BadRequest,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: "x-incorrect-routing-strategy", RawValue: []byte(routingStrategy),
			}}}, ""), utils.User{}, routingStrategy
	}

	if username != "" {
		user, err = utils.GetUser(utils.User{Name: username}, s.redisClient)
		if err != nil {
			klog.ErrorS(err, "unable to process user info", "requestID", requestID, "username", username)
			return buildErrorResponse(InternalServerError, err.Error(), "x-error-get-user", "true"), utils.User{}, routingStrategy
		}

		errRes, err = s.checkLimits(ctx, user)
		if errRes != nil {
			klog.ErrorS(err, "error on checking limits", "requestID", requestID, "username", username)
			return errRes, utils.User{}, routingStrategy
		}
	}

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: []*configPb.HeaderValueOption{
							{
								Header: &configPb.HeaderValue{
									Key:      "x-went-into-req-headers",
									RawValue: []byte("true"),
								},
							},
						},
					},
					ClearRouteCache: true,
				},
			},
		},
	}, user, routingStrategy
}

func (s *Server) HandleRequestBody(ctx context.Context, requestID string, req *extProcPb.ProcessingRequest, user utils.User, routingStrategy string) (*extProcPb.ProcessingResponse, string) {
	klog.Info("--- In RequestBody processing")
	var model, targetPodIP string
	var ok bool
	var jsonMap map[string]interface{}

	body := req.Request.(*extProcPb.ProcessingRequest_RequestBody)
	if err := json.Unmarshal(body.RequestBody.GetBody(), &jsonMap); err != nil {
		return generateErrorResponse(envoyTypePb.StatusCode_InternalServerError,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: "x-request-body-processing-error", RawValue: []byte("true")}}},
			"error processing request body"), targetPodIP
	}

	if model, ok = jsonMap["model"].(string); !ok || model == "" || !s.cache.CheckModelExists(model) {
		return generateErrorResponse(envoyTypePb.StatusCode_InternalServerError,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: "x-no-model", RawValue: []byte(model)}}},
			fmt.Sprintf("no model in request body or model %s does not exist", model)), targetPodIP
	}

	headers := []*configPb.HeaderValueOption{}
	switch {
	case routingStrategy == "":
		buildEnvoyProxyHeaders(headers, "model", model)
		klog.InfoS("request start", "requestID", requestID, "model", model)
	case routingStrategy != "":
		pods, err := s.cache.GetPodsForModel(model)
		if len(pods) == 0 || err != nil {
			return buildErrorResponse(InternalServerError,
				fmt.Sprintf("error on getting pods for model %s", model),
				"x-no-model-deployment", "true"), targetPodIP
		}
		targetPodIP, err = s.routers[routingStrategy].Route(ctx, pods)
		if err != nil {
			return buildErrorResponse(InternalServerError,
				fmt.Sprintf("error on selecting target pod for routingStrategy: %s", routingStrategy),
				"x-error-routing", "true"), targetPodIP
		}

		buildEnvoyProxyHeaders(headers, "routing-strategy", routingStrategy, "target-pod", targetPodIP)
		klog.InfoS("request start", "requestID", requestID, "model", model, "routingStrategy", routingStrategy, "targetPodIP", targetPodIP)
	}

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: headers,
					},
				},
			},
		},
	}, targetPodIP
}

func (s *Server) HandleResponseBody(ctx context.Context, requestID string, req *extProcPb.ProcessingRequest, user utils.User, targetPodIP string) *extProcPb.ProcessingResponse {
	klog.Infof("--- In ResponseBody processing")
	b := req.Request.(*extProcPb.ProcessingRequest_ResponseBody)

	var res openai.CompletionResponse
	if err := json.Unmarshal(b.ResponseBody.Body, &res); err != nil {
		klog.ErrorS(err, "error to unmarshal response", "requestID", requestID)
		return buildErrorResponse(InternalServerError, err.Error(), "x-error-response-unmarshal", "true")
	}

	defer func() {
		go func() {
			s.cache.AddRequestTrace(res.Model, res.Usage.PromptTokens, res.Usage.CompletionTokens)
		}()
	}()

	headers := []*configPb.HeaderValueOption{}
	if user.Name != "" {
		s.mu.Lock()
		rpm, err := s.getRPM(ctx, user.Name)
		if err != nil {
			return buildErrorResponse(envoyTypePb.StatusCode_InternalServerError, err.Error(), "x-error-get-rpm", "true")
		}

		tpm, err := s.ratelimiter.Incr(ctx, fmt.Sprintf("%v_TPM_CURRENT", user), int64(res.Usage.TotalTokens))
		if err != nil {
			return buildErrorResponse(envoyTypePb.StatusCode_InternalServerError, err.Error(), "x-error-update-tpm", "true")
		}
		s.mu.Unlock()

		headers = buildEnvoyProxyHeaders(headers, "x-updated-rpm", string(rpm), "x-updated-tpm", string(tpm))
		klog.InfoS("request end", "requestID", requestID, "rpm", rpm, "tpm", tpm)
	}
	if targetPodIP != "" {
		headers = buildEnvoyProxyHeaders(headers, "target-pod", targetPodIP)
	}

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseBody{
			ResponseBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: headers,
					},
				},
			},
		},
	}
}

func (s *Server) checkLimits(ctx context.Context, user utils.User) (*extProcPb.ProcessingResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if user.Rpm == 0 {
		user.Rpm = int64(DefaultRPM)
	}
	if user.Tpm == 0 {
		user.Tpm = user.Rpm * int64(DefaultTPMMultiplier)
	}

	code, err := s.checkRPM(ctx, user.Name, user.Rpm)
	if err != nil {
		return buildErrorResponse(code, err.Error(), "x-rpm-exceeded", "true"), err
	}

	code, err = s.checkTPM(ctx, user.Name, user.Tpm)
	if err != nil {
		return buildErrorResponse(code, err.Error(), "x-tpm-exceeded", "true"), err
	}

	code, err = s.incrRPM(ctx, user.Name)
	if err != nil {
		return buildErrorResponse(code, err.Error(), "x-error-incr-rpm", "true"), err
	}

	return nil, nil
}

func (s *Server) getRPM(ctx context.Context, username string) (int64, error) {
	return s.ratelimiter.Get(ctx, fmt.Sprintf("%v_RPM_CURRENT", username))
}

func (s *Server) checkRPM(ctx context.Context, username string, rpmLimit int64) (envoyTypePb.StatusCode, error) {
	rpmCurrent, err := s.getRPM(ctx, username)
	if err != nil {
		return envoyTypePb.StatusCode_InternalServerError, fmt.Errorf("fail to get current RPM for user: %v", username)
	}

	if rpmCurrent >= rpmLimit {
		return envoyTypePb.StatusCode_TooManyRequests, fmt.Errorf("user: %v has exceeded RPM_LIMIT: %v", username, rpmLimit)
	}

	return envoyTypePb.StatusCode_OK, nil
}

func (s *Server) incrRPM(ctx context.Context, username string) (envoyTypePb.StatusCode, error) {
	_, err := s.ratelimiter.Incr(ctx, fmt.Sprintf("%v_RPM_CURRENT", username), 1)
	if err != nil {
		return envoyTypePb.StatusCode_InternalServerError, fmt.Errorf("fail to increment current RPM for user: %v", username)
	}

	return envoyTypePb.StatusCode_OK, nil
}

func (s *Server) checkTPM(ctx context.Context, username string, tpmLimit int64) (envoyTypePb.StatusCode, error) {
	tpmCurrent, err := s.ratelimiter.Get(ctx, fmt.Sprintf("%v_TPM_CURRENT", username))
	if err != nil {
		return envoyTypePb.StatusCode_InternalServerError, fmt.Errorf("fail to get current TPM for user: %v", username)
	}

	if tpmCurrent >= tpmLimit {
		return envoyTypePb.StatusCode_TooManyRequests, fmt.Errorf("user: %v has exceeded TPM_LIMIT: %v", username, tpmLimit)
	}

	return envoyTypePb.StatusCode_OK, nil
}

func generateErrorResponse(statusCode envoyTypePb.StatusCode, headers []*configPb.HeaderValueOption, body string) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{
					Code: statusCode,
				},
				Headers: &extProcPb.HeaderMutation{
					SetHeaders: headers,
				},
				Body: body,
			},
		},
	}
}
