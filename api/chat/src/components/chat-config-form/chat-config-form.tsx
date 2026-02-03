import {ReactElement, useState} from 'react';
import {ChatConfigInterface} from '@/utils/interfaces';
import {Button} from '../ui/button';
import {Input} from '../ui/input';
import {Label} from '../ui/label';
import {Textarea} from '../ui/textarea';
import {Switch} from '../ui/switch';
import {DialogDescription, DialogHeader, DialogTitle} from '../ui/dialog';
import {spaceClick} from '@/utils/methods';

interface ChatConfigFormProps {
	onSubmit: (config: ChatConfigInterface) => void;
	onBack?: () => void;
	onCancel: () => void;
	hideBackButton?: boolean;
	defaultCustomerId?: string;
}

const ChatConfigForm = ({onSubmit, onBack, onCancel, hideBackButton = false, defaultCustomerId = ''}: ChatConfigFormProps): ReactElement => {
	const [config, setConfig] = useState<ChatConfigInterface>({
		tenant_id: '',
		chatbot_id: '',
		customer_id: defaultCustomerId,
		session_id: '',
		md5_checksum: '',
		is_preview: false,
		preview_action_book_ids: [],
		autofill_params: {},
		timeout: 60,
	});

	const [actionBookIdsInput, setActionBookIdsInput] = useState('');
	const [autofillParamsInput, setAutofillParamsInput] = useState('');
	const [showAdvanced, setShowAdvanced] = useState(false);

	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		
		// 解析 action book IDs
		const actionBookIds = actionBookIdsInput
			.split(',')
			.map(id => id.trim())
			.filter(id => id.length > 0);
		
		// 解析 autofill params
		let autofillParams = {};
		if (autofillParamsInput.trim()) {
			try {
				autofillParams = JSON.parse(autofillParamsInput);
			} catch (e) {
				alert('Autofill params must be valid JSON');
				return;
			}
		}

		onSubmit({
			...config,
			preview_action_book_ids: actionBookIds.length > 0 ? actionBookIds : undefined,
			autofill_params: Object.keys(autofillParams).length > 0 ? autofillParams : undefined,
		});
	};

	const handleChange = (field: keyof ChatConfigInterface, value: any) => {
		setConfig(prev => ({...prev, [field]: value}));
	};

	return (
		<div className='h-full flex flex-col'>
			<DialogHeader>
				<DialogTitle>
					<div className='mb-[12px] mt-[24px] w-full flex justify-between items-center ps-[30px] pe-[20px]'>
						<DialogDescription className='text-[20px] font-semibold'>Chat Configuration</DialogDescription>
						<img
							role='button'
							tabIndex={0}
							onKeyDown={spaceClick}
							onClick={onCancel}
							className='cursor-pointer rounded-full'
							src='icons/close.svg'
							alt='close'
							height={24}
							width={24}
						/>
					</div>
				</DialogTitle>
			</DialogHeader>

			<form onSubmit={handleSubmit} className='flex flex-col flex-1 overflow-hidden'>
				<div className='flex-1 overflow-auto px-[30px] pb-4'>
					<div className='space-y-4'>
						{/* Required Fields */}
						<div className='space-y-2'>
							<Label htmlFor='tenant_id' className='text-sm font-medium'>
								Tenant ID <span className='text-red-500'>*</span>
							</Label>
							<Input
								id='tenant_id'
								value={config.tenant_id}
								onChange={(e) => handleChange('tenant_id', e.target.value)}
								placeholder='Enter tenant ID'
								required
								className='w-full'
							/>
						</div>

						<div className='space-y-2'>
							<Label htmlFor='chatbot_id' className='text-sm font-medium'>
								Chatbot ID <span className='text-red-500'>*</span>
							</Label>
							<Input
								id='chatbot_id'
								value={config.chatbot_id}
								onChange={(e) => handleChange('chatbot_id', e.target.value)}
								placeholder='Enter chatbot ID'
								required
								className='w-full'
							/>
						</div>

						<div className='space-y-2'>
							<Label htmlFor='customer_id' className='text-sm font-medium'>
								Customer ID <span className='text-red-500'>*</span>
							</Label>
							<Input
								id='customer_id'
								value={config.customer_id}
								onChange={(e) => handleChange('customer_id', e.target.value)}
								placeholder='Enter customer ID'
								required
								className='w-full'
							/>
						</div>

						<div className='space-y-2'>
							<Label htmlFor='session_id' className='text-sm font-medium'>
								Session ID <span className='text-red-500'>*</span>
							</Label>
							<Input
								id='session_id'
								value={config.session_id}
								onChange={(e) => handleChange('session_id', e.target.value)}
								placeholder='Enter session ID'
								required
								className='w-full'
							/>
						</div>


						<div className='space-y-2'>
							<Label htmlFor='session_title' className='text-sm font-medium'>
								Session Title
							</Label>
							<Input
								id='session_title'
								value={config.session_title}
								onChange={(e) => handleChange('session_title', e.target.value)}
								placeholder='Enter session title'
								className='w-full'
							/>
						</div>

						{/* Advanced Options Toggle */}
						<div className='flex items-center space-x-2 pt-2 pb-2 border-t border-gray-200'>
							<Switch
								id='show_advanced'
								checked={showAdvanced}
								onCheckedChange={setShowAdvanced}
							/>
							<Label htmlFor='show_advanced' className='text-sm font-medium cursor-pointer'>
								Show Advanced Options
							</Label>
						</div>

						{/* Advanced Fields */}
						{showAdvanced && (
							<div className='space-y-4 border-t pt-4'>
								<div className='space-y-2'>
									<Label htmlFor='md5_checksum' className='text-sm font-medium'>
										MD5 Checksum
									</Label>
									<Input
										id='md5_checksum'
										value={config.md5_checksum}
										onChange={(e) => handleChange('md5_checksum', e.target.value)}
										placeholder='Enter MD5 checksum'
										className='w-full'
									/>
									<p className='text-xs text-gray-500'>
										Optional: MD5 checksum for agent configuration validation
									</p>
								</div>

								<div className='space-y-2'>
									<div className='flex items-center space-x-2'>
										<Switch
											id='is_preview'
											checked={config.is_preview || false}
											onCheckedChange={(checked) => handleChange('is_preview', checked)}
										/>
										<Label htmlFor='is_preview' className='text-sm font-medium cursor-pointer'>
											Preview Mode
										</Label>
									</div>
									<p className='text-xs text-gray-500'>
										Enable to preview specific action books
									</p>
								</div>

								<div className='space-y-2'>
									<Label htmlFor='preview_action_book_ids' className='text-sm font-medium'>
										Preview Action Book IDs
									</Label>
									<Input
										id='preview_action_book_ids'
										value={actionBookIdsInput}
										onChange={(e) => setActionBookIdsInput(e.target.value)}
										placeholder='Enter IDs separated by commas'
										disabled={!config.is_preview}
										className='w-full'
									/>
									<p className='text-xs text-gray-500'>
										Comma-separated list of action book IDs to preview
									</p>
								</div>

								<div className='space-y-2'>
									<Label htmlFor='autofill_params' className='text-sm font-medium'>
										Autofill Parameters (JSON)
									</Label>
									<Textarea
										id='autofill_params'
										value={autofillParamsInput}
										onChange={(e) => setAutofillParamsInput(e.target.value)}
										placeholder='{"key": "value"}'
										className='w-full font-mono text-sm'
										rows={3}
									/>
									<p className='text-xs text-gray-500'>
										Optional: JSON object with parameters to auto-fill
									</p>
								</div>

								<div className='space-y-2'>
									<Label htmlFor='timeout' className='text-sm font-medium'>
										Timeout (seconds)
									</Label>
									<Input
										id='timeout'
										type='number'
										value={config.timeout}
										onChange={(e) => handleChange('timeout', parseInt(e.target.value) || 60)}
										placeholder='60'
										min={1}
										max={300}
										className='w-full'
									/>
									<p className='text-xs text-gray-500'>
										Maximum wait time for AI response (1-300 seconds)
									</p>
								</div>
							</div>
						)}
					</div>
				</div>

				{/* Action Buttons */}
				<div className='flex justify-between gap-3 px-[30px] pb-[24px] pt-4 border-t border-gray-200'>
					{!hideBackButton && onBack && (
						<Button
							type='button'
							variant='outline'
							onClick={onBack}
							className='flex-1'
						>
							← Back
						</Button>
					)}
					<Button
						type='submit'
						className={hideBackButton ? 'w-full' : 'flex-1'} 
						style={{backgroundColor: '#3C8C71'}}
						onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#2d6a56'}
						onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#3C8C71'}
						disabled={!config.tenant_id || !config.chatbot_id || !config.customer_id || !config.session_id}
					>
						Start Chat →
					</Button>
				</div>
			</form>
		</div>
	);
};

export default ChatConfigForm;

